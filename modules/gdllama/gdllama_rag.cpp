
#include "gdllama_rag.h"

#include "llama-win/common.h"
#include "llama-win/llama.h"

#include "sqlite-vec.h"

static bool didInstantiate = false;
static gpt_params params;
static llama_model *model;
static llama_context *ctx;
static sqlite3 *db;

static void batch_add_seq(llama_batch &batch, const std::vector<int32_t> &tokens, llama_seq_id seq_id) {
	size_t n_tokens = tokens.size();
	for (size_t i = 0; i < n_tokens; i++) {
		llama_batch_add(batch, tokens[i], i, { seq_id }, true);
	}
}
static void batch_decode(llama_context *ctx_, llama_batch &batch, float *output, int n_seq, int n_embd) {
	// clear previous kv_cache values (irrelevant for embeddings)
	llama_kv_cache_clear(ctx_);

	// run model
	fprintf(stderr, "%s: n_tokens = %d, n_seq = %d\n", __func__, batch.n_tokens, n_seq);
	if (llama_decode(ctx_, batch) < 0) {
		fprintf(stderr, "%s : failed to decode\n", __func__);
	}

	for (int i = 0; i < batch.n_tokens; i++) {
		if (!batch.logits[i]) {
			continue;
		}

		// try to get sequence embeddings - supported only when pooling_type is not NONE
		const float *embd = llama_get_embeddings_seq(ctx_, batch.seq_id[i][0]);

		if (embd == NULL) {
			embd = llama_get_embeddings_ith(ctx, i);
			if (embd == NULL) {
				fprintf(stderr, "%s: failed to get embeddings for token %d\n", __func__, i);
				continue;
			}
		}

		float *out = output + batch.seq_id[i][0] * n_embd;
		//TODO: I would also add a parameter here to enable normalization or not.
		/*fprintf(stdout, "unnormalized_embedding:");
		for (int hh = 0; hh < n_embd; hh++) {
			fprintf(stdout, "%9.6f ", embd[hh]);
		}
		fprintf(stdout, "\n");*/
		llama_embd_normalize(embd, out, n_embd);
	}
}

void LlmDBMetaData::_bind_methods() {
}

LlmDBMetaData::LlmDBMetaData() :
		data_name{ "default_name" },
		data_type{ 0 } {}

LlmDBMetaData::~LlmDBMetaData() {}

Ref<LlmDBMetaData> LlmDBMetaData::create(String data_name, int data_type) {
	Ref<LlmDBMetaData> data = memnew(LlmDBMetaData());
	data->set_data_name(data_name);
	data->set_data_type(data_type);
	return data;
}

Ref<LlmDBMetaData> LlmDBMetaData::create_int(String data_name) {
	Ref<LlmDBMetaData> data = memnew(LlmDBMetaData());
	data->set_data_name(data_name);
	data->set_data_type(0);
	return data;
}

Ref<LlmDBMetaData> LlmDBMetaData::create_real(String data_name) {
	Ref<LlmDBMetaData> data = memnew(LlmDBMetaData());
	data->set_data_name(data_name);
	data->set_data_type(1);
	return data;
}

Ref<LlmDBMetaData> LlmDBMetaData::create_text(String data_name) {
	Ref<LlmDBMetaData> data = memnew(LlmDBMetaData());
	data->set_data_name(data_name);
	data->set_data_type(2);
	return data;
}

Ref<LlmDBMetaData> LlmDBMetaData::create_blob(String data_name) {
	Ref<LlmDBMetaData> data = memnew(LlmDBMetaData());
	data->set_data_name(data_name);
	data->set_data_type(2);
	return data;
}

String LlmDBMetaData::get_data_name() const {
	return data_name;
};

void LlmDBMetaData::set_data_name(const String p_data_name) {
	data_name = p_data_name;
};

int LlmDBMetaData::get_data_type() const {
	return data_type;
}

void LlmDBMetaData::set_data_type(const int p_data_type) {
	data_type = p_data_type;
}

String Rag::type_int_to_string(int meta_data_type) {
	switch (meta_data_type) {
		case LlmDBMetaDataType::INTEGER:
			return "INT";
		case LlmDBMetaDataType::REAL:
			return "REAL";
		case LlmDBMetaDataType::TEXT:
			return "TEXT";
		case LlmDBMetaDataType::BLOB:
			return "";
		default: {
			printf("Wrong meta type: %d\n", meta_data_type);
			return "";
		}
	}
}

Variant::Type Rag::type_int_to_variant(int meta_data_type) {
	switch (meta_data_type) {
		case LlmDBMetaDataType::INTEGER:
			return Variant::INT;
		case LlmDBMetaDataType::REAL:
			return Variant::FLOAT;
		case LlmDBMetaDataType::TEXT:
			return Variant::STRING;
		case LlmDBMetaDataType::BLOB:
			return Variant::VARIANT_MAX;
		default: {
			printf("Wrong meta type for variant: %d\n", meta_data_type);
			return Variant::NIL;
		}
	}
}

Rag::Rag() {
	if (didInstantiate) {
		printf("There can be only one Rag instance\n");
		return;
	} else
		didInstantiate = true;

	int rc = SQLITE_OK;
	rc = sqlite3_auto_extension((void (*)())sqlite3_vec_init);
	if (rc != SQLITE_OK) {
		printf("Unable to load sqlite3_vec extension\n");
	}

	params.n_threads = cpu_get_num_math();
	if (params.n_threads > 8)
		params.n_threads = 8;
	params.n_ctx = 512;

	//params.use_mmap = false;

	absolute_separators = PackedStringArray();
	absolute_separators.append("\n\n");
	absolute_separators.append("\n");

	chunk_separators = PackedStringArray();
	chunk_separators.append(".");
	chunk_separators.append(",");
	chunk_separators.append(String::utf8("\uff0c"));
	chunk_separators.append(String::utf8("\u3001"));
	chunk_separators.append(String::utf8("\uff0e"));
	chunk_separators.append(String::utf8("\u3002"));
	chunk_separators.append("\u200b");
	chunk_separators.append(" ");
	chunk_separators.append("");

	chunk_size = 100;
	chunk_overlap = 20;
	embedding_size = 384;

	meta = TypedArray<LlmDBMetaData>();
	meta.append(LlmDBMetaData::create_text("id"));
	meta.append(LlmDBMetaData::create_int("type"));
}
Rag::~Rag() {
    printf("Rag dealloc\n");
}

void Rag::openDB() {
	if (db != nullptr) {
		printf("Cannot open db, db is already open\n");
		return;
	}

	int rc = SQLITE_OK;
	rc = sqlite3_open_v2(dbPath.c_str(), &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_URI, nullptr); //SQLITE_OPEN_READONLY
	if (rc != SQLITE_OK) {
		printf("Failed to open database: %s\n", sqlite3_errmsg(db));
	}

	printf("openDB -- done\n");
}
void Rag::closeDB() {
	if (db != nullptr) {
		int rc = SQLITE_OK;
		rc = sqlite3_close_v2(db);
		db = nullptr;
		if (rc != SQLITE_OK) {
			printf("Failed to close database\n");
		}
	} else {
		printf("Cannot close db, no db is open\n");
	}

	printf("closeDB -- done\n");
}

void Rag::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_path", "value"), &Rag::set_path, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("set_db_path", "value"), &Rag::set_db_path, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("initialize"), &Rag::initialize);
	ClassDB::bind_method(D_METHOD("retrieve_similar_texts", "text", "where", "n_results"), &Rag::retrieve_similar_texts, DEFVAL(""), DEFVAL(""), DEFVAL(3));
	ClassDB::bind_method(D_METHOD("create_db"), &Rag::create_db);
	ClassDB::bind_method(D_METHOD("insert_text", "text", "meta_dict"), &Rag::insert_text, DEFVAL(""), DEFVAL(NULL));
	ClassDB::bind_method(D_METHOD("split_text", "text"), &Rag::split_text, DEFVAL(""));
}

void Rag::set_path(const String &modelPath) {
	params.model = std::string(modelPath.utf8().get_data());
}
void Rag::set_db_path(const String &dbPath_) {
	dbPath = std::string(dbPath_.utf8().get_data());
}

void Rag::initialize() {
	if (model != NULL) {
		printf("Rag is initialized already\n");
		return;
	}

	openDB();

	params.embedding = true;
	// For non-causal models, batch size must be equal to ubatch size
	params.n_ubatch = params.n_batch;

	if (params.seed == LLAMA_DEFAULT_SEED) {
		params.seed = time(NULL);
	}

	fprintf(stderr, "%s: seed  = %u\n", __func__, params.seed);

	std::mt19937 rng(params.seed);

	llama_backend_init();
	llama_numa_init(params.numa);

	// load the model
	llama_init_result llama_init = llama_init_from_gpt_params(params);
	model = llama_init.model;

	if (model == NULL) {
		fprintf(stderr, "%s: error: unable to load model\n", __func__);
		return;
	}

	ctx = llama_init.context;
	if (ctx == NULL) {
		fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, params.model.c_str());
		llama_free_model(model);
		if (db != nullptr) closeDB();
	} else {
		llama_free(ctx);
		ctx = NULL;
		embedding_size = llama_n_embd(model);
	}

	llama_backend_free();
}

PackedStringArray Rag::retrieve_similar_texts(const String text, const String where, const int n_results) {
	printf("retrieve_similar_texts\n");
	std::vector<float> embedding = compute_embedding(std::string(text.utf8().get_data()));

	String statement_filter = "SELECT rowid FROM my_table";
	if (!where.is_empty()) {
		statement_filter += " WHERE " + where;
	}

	String statement_virtual = "SELECT rowid FROM my_table_virtual WHERE text MATCH '[";

	for (float f : embedding) {
		statement_virtual += String::num_real(f) + ", ";
	}

	statement_virtual = statement_virtual.trim_suffix(", ");
	statement_virtual += "]' ";

	statement_virtual += "AND rowid in (" + statement_filter + ") ";

	statement_virtual += "ORDER BY distance LIMIT " + String::num_int64(n_results);

	String statement_final = "SELECT id FROM my_table WHERE rowid IN (" + statement_virtual + ");";

	//printf("retrieve_text_array statement: %s\n", statement_final.utf8().get_data());

	PackedStringArray array{ PackedStringArray() };

	int rc = SQLITE_OK;
	sqlite3_stmt *stmt;
	sqlite3_prepare_v2(db, statement_final.utf8().get_data(), -1, &stmt, NULL);
	if (rc != SQLITE_OK) {
		printf("Failed to prepare statement\n");
		return array;
	}

	while (true) {
		rc = sqlite3_step(stmt);
		if (rc == SQLITE_DONE)
			break;
		if (rc != SQLITE_ROW) {
			printf("Error: %s\n", sqlite3_errmsg(db));
			return array;
		}
		const char *c = (char *)sqlite3_column_text(stmt, 0);

		String retrieved_text = String::utf8(c);
		//printf("retrieved text: %s\n", retrieved_text.utf8().get_data());
		array.append(retrieved_text);
	}

	sqlite3_finalize(stmt);

	printf("retrieve_similar_texts -- done\n");

	return array;
}
bool Rag::create_db() {
	printf("create_db:\n");
	String statement = "CREATE TABLE IF NOT EXISTS my_table (";
	for (int i = 0; i < meta.size(); i++) {
		Ref<LlmDBMetaData> sd = Object::cast_to<LlmDBMetaData>(meta[i]);
		statement += "'" + sd->get_data_name() + "' ";
		statement += type_int_to_string(sd->get_data_type());
		statement += ", ";
	}

	statement += String("text") + " float[" + String::num_int64(embedding_size) + "]";

	statement += ");";

	printf("Create table statement: %s\n", statement.utf8().get_data());

	if(!execute(statement)) return false;

	// Also create a table to store meta data
	String meta_table_name = "my_table_meta";

	String statement_meta = "CREATE TABLE IF NOT EXISTS " + meta_table_name + " (";
	for (int i = 0; i < meta.size(); i++) {
		Ref<LlmDBMetaData> sd = Object::cast_to<LlmDBMetaData>(meta[i]);
		statement_meta += " '" + sd->get_data_name() + "' ";
		statement_meta += type_int_to_string(sd->get_data_type());
		if (i == 0) {
			statement_meta += " PRIMARY KEY";
		}

		statement_meta += ", ";
	}

	statement_meta = statement_meta.trim_suffix(", ");
	statement_meta += ") WITHOUT ROWID;";

	printf("Create meta table statement: %s\n", statement_meta.utf8().get_data());

	if(!execute(statement_meta)) return false;

	String statement_virtual = "CREATE VIRTUAL TABLE IF NOT EXISTS my_table_virtual USING vec0(text float[" + String::num_int64(embedding_size) + "]);";

	printf("Create virtual table statement: %s\n", statement_virtual.utf8().get_data());

	return execute(statement_virtual);
}
bool Rag::insert_text(const String text, const Dictionary meta_dict) {
	printf("insert_text_by_meta\n");

	std::vector<float> embedding = compute_embedding(text.utf8().get_data());

	String statement_1 = "INSERT INTO my_table";
	String statement_2 = "(";
	String statement_3 = "(";
	Dictionary p_meta_dict = meta_dict.duplicate(false);
	PackedStringArray array_bind{ PackedStringArray() };
	for (int i = 0; i < meta.size(); i++) {
		Ref<LlmDBMetaData> sd = Object::cast_to<LlmDBMetaData>(meta[i]);
		if (p_meta_dict.has(sd->get_data_name())) {
			Variant v = p_meta_dict.get(sd->get_data_name(), NULL);
			if (v.get_type() != type_int_to_variant(sd->get_data_type())) {
				printf("Wrong data type for key %s\n", (sd->get_data_name() + " : " + v.get_type_name(v.get_type()) + " instead of " + sd->get_data_type()).utf8().get_data());
			}

			p_meta_dict.erase(sd->get_data_name());

			switch (sd->get_data_type()) {
				case LlmDBMetaDataType::INTEGER: {
					statement_2 += sd->get_data_name() + ", ";
					int k = v;
					statement_3 += String::num_int64(k) + ", ";
					break;
				}
				case LlmDBMetaDataType::REAL: {
					statement_2 += sd->get_data_name() + ", ";
					float f = v;
					statement_3 += String::num_real(f) + ", ";
					break;
				}
				case LlmDBMetaDataType::TEXT: {
					statement_2 += sd->get_data_name() + ", ";
					String s = v;
					statement_3 += "?, ";
					array_bind.append(s);
					break;
				}
				case LlmDBMetaDataType::BLOB: {
					statement_2 += sd->get_data_name() + ", ";
					String s = v.stringify();
					statement_3 += "?, ";
					array_bind.append(s);
					break;
				}
			}
		}
	}

	if (!p_meta_dict.is_empty()) {
		printf("Some meta data has incorrect key or incorrect format, key: %s\n", JSON::stringify(p_meta_dict.keys()).utf8().get_data());
	}

	statement_2 = statement_2 + "text)";
	statement_3 = statement_3 + "'[";
	array_bind.append(text);

	for (float f : embedding) {
		statement_3 += String::num_real(f) + ", ";
	}

	statement_3 = statement_3.trim_suffix(", ");
	statement_3 += "]')";

	String statement = statement_1 + " " + statement_2 + " VALUES " + statement_3 + ";";
	//printf("insert_text_by_meta statement: %s\n", statement.utf8().get_data());

	int rc = SQLITE_OK;
	sqlite3_stmt *stmt;
	rc = sqlite3_prepare_v2(db, statement.utf8().get_data(), -1, &stmt, nullptr);
	if (rc != SQLITE_OK) {
		printf("Failed to perpare statement\n");
		return false;
	}
	for (int i = 0; i < array_bind.size(); i++) {
		sqlite3_bind_text(stmt, i + 1, array_bind[i].utf8().get_data(), -1, SQLITE_TRANSIENT);
	}
	rc = sqlite3_step(stmt);
	if (rc != SQLITE_DONE) {
		printf("Error: %s\n", String::utf8(sqlite3_errmsg(db)).utf8().get_data());
		return false;
	}
	sqlite3_finalize(stmt);

	String statement_virtual = "INSERT OR REPLACE INTO my_table_virtual (rowid, text) SELECT rowid, text FROM my_table WHERE rowid=last_insert_rowid()";
	//printf("insert_text virtual statement: %s\n", statement_virtual.utf8().get_data());
	return execute(statement_virtual);
}
PackedStringArray Rag::split_text(const String text) {
	if (text.is_empty()) {
		return PackedStringArray();
	}

	if (absolute_separators.is_empty()) {
		printf("Empty absolute_separators\n");
	}

	PackedStringArray separated_array_a = absolute_split_text(text, 0);

	if (chunk_separators.is_empty()) {
		printf("Empty chunk_separators\n");
		return PackedStringArray();
	}

	PackedStringArray array = PackedStringArray();

	for (String s : separated_array_a) {
		array.append_array(chunk_split_text(s, 0));
	}

	return array;
};

int print_all_callback(void *unused, int count, char **data, char **columns) {
	printf("print_all_callback called\n");

	printf("There are %d column(s)\n", count);

	for (int idx = 0; idx < count; idx++) {
		std::string column = std::string(columns[idx]);
		std::string columnData = std::string(data[idx]);
		printf("The data in column %s is: %s\n", column.c_str(), columnData.c_str());
	}

	return 0;
}
bool Rag::execute(String statement) {
	int rc = SQLITE_OK;
	char *errmsg;
	rc = sqlite3_exec(db, statement.utf8().get_data(), print_all_callback, nullptr, &errmsg);
	if (rc != SQLITE_OK) return false;
	else return true;
}

std::vector<float> Rag::compute_embedding(
		std::string prompt
		//, std::function<void(std::vector<float>)> on_compute_finished
) {
	params.prompt = prompt;

	struct llama_context_params lparams = llama_context_params_from_gpt_params(params);
	ctx = llama_new_context_with_model(model, lparams);

	const int n_ctx_train = llama_n_ctx_train(model);
	const int n_ctx = llama_n_ctx(ctx);

	const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
	if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
		fprintf(stderr, "%s: error: pooling type NONE not supported\n", __func__);
		return std::vector<float>{};
	}

	if (n_ctx > n_ctx_train) {
		fprintf(stderr, "%s: warning: model was trained on only %d context tokens (%d specified)\n",
				__func__, n_ctx_train, n_ctx);
	}

	// print system information
	{
		fprintf(stderr, "\n");
		fprintf(stderr, "%s\n", gpt_params_get_system_info(params).c_str());
	}

	// split the prompt into lines
	//std::vector<std::string> prompts = split_lines(params.prompt);
	std::vector<std::string> prompts = std::vector<std::string>{ params.prompt };

	// max batch size
	const uint64_t n_batch = params.n_batch;
	//GGML_ASSERT(params.n_batch >= params.n_ctx);

	// tokenize the prompts and trim
	std::vector<std::vector<int32_t>> inputs;
	for (const auto &prompt : prompts) {
		auto inp = ::llama_tokenize(ctx, prompt, true, false);
		if (inp.size() > n_batch) {
			fprintf(stderr, "%s: error: number of tokens in input line (%lld) exceeds batch size (%lld), increase batch size and re-run\n",
					__func__, (long long int)inp.size(), (long long int)n_batch);
			//on_compute_finished(std::vector<float>{});
			return std::vector<float>{};
		}
		// add eos if not present
		if (llama_token_eos(model) >= 0 && (inp.empty() || inp.back() != llama_token_eos(model))) {
			inp.push_back(llama_token_eos(model));
		}
		inputs.push_back(inp);
	}

	// check if the last token is SEP
	// it should be automatically added by the tokenizer when 'tokenizer.ggml.add_eos_token' is set to 'true'
	for (auto &inp : inputs) {
		if (inp.empty() || inp.back() != llama_token_sep(model)) {
			fprintf(stderr, "%s: warning: last token in the prompt is not SEP\n", __func__);
			fprintf(stderr, "%s:          'tokenizer.ggml.add_eos_token' should be set to 'true' in the GGUF header\n", __func__);
		}
	}

	// tokenization stats
	if (params.verbose_prompt) {
		for (int i = 0; i < (int)inputs.size(); i++) {
			fprintf(stderr, "%s: prompt %d: '%s'\n", __func__, i, prompts[i].c_str());
			fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, inputs[i].size());
			for (int j = 0; j < (int)inputs[i].size(); j++) {
				fprintf(stderr, "%6d -> '%s'\n", inputs[i][j], llama_token_to_piece(ctx, inputs[i][j]).c_str());
			}
			fprintf(stderr, "\n\n");
		}
	}

	// initialize batch
	const int n_prompts = prompts.size();
	struct llama_batch batch = llama_batch_init(n_batch, 0, 1);

	// allocate output
	const int n_embd = llama_n_embd(model);
	std::vector<float> embeddings(n_prompts * n_embd, 0);
	float *emb = embeddings.data();

	// break into batches
	int p = 0; // number of prompts processed already
	int s = 0; // number of prompts in current batch
	for (int k = 0; k < n_prompts; k++) {
		// clamp to n_batch tokens
		auto &inp = inputs[k];

		const uint64_t n_toks = inp.size();

		// encode if at capacity
		if (batch.n_tokens + n_toks > n_batch) {
			float *out = emb + p * n_embd;
			batch_decode(ctx, batch, out, s, n_embd);
			llama_batch_clear(batch);
			p += s;
			s = 0;
		}

		// add to batch
		batch_add_seq(batch, inp, s);
		s += 1;
	}

	// final batch
	float *out = emb + p * n_embd;
	batch_decode(ctx, batch, out, s, n_embd);

	// print the first part of the embeddings or for a single prompt, the full embedding
	/*fprintf(stdout, "\n");
	for (int j = 0; j < n_prompts; j++) {
		fprintf(stdout, "embedding %d: ", j);
		for (int i = 0; i < (n_prompts > 1 ? std::min(16, n_embd) : n_embd); i++) {
			fprintf(stdout, "%9.6f ", emb[j * n_embd + i]);
		}
		fprintf(stdout, "\n");
	}*/

	// clean up
	llama_print_timings(ctx);
	llama_batch_free(batch);
	llama_kv_cache_clear(ctx);
	llama_reset_timings(ctx);
	if (ctx) {
		llama_free(ctx);
		ctx = NULL;
	}
	//llama_free_model(model);
	llama_backend_free();

	//on_compute_finished(embeddings);

	return embeddings;
}

float Rag::similarity_cos(std::vector<float> embd1, std::vector<float> embd2) {
	if (embd1.size() != embd2.size()) {
		printf("Error: embedding sizes don't match\n");
		return 0.0;
	}

	return llama_embd_similarity_cos(embd1.data(), embd2.data(), embd1.size());
}
/*float Rag::similarity_cos(std::vector<float> array1, std::vector<float> array2) {
	if (array1.size() != array2.size() || array1.size() == 0) {
		printf("Error: embedding sizes don't match");
		return 0.0;
	}

	double sum = 0.0;
	double sum1 = 0.0;
	double sum2 = 0.0;

	for (int i = 0; i < array1.size(); i++) {
		sum += array1[i] * array2[i];
		sum1 += array1[i] * array1[i];
		sum2 += array2[i] * array2[i];
	}

	return sum / (sqrt(sum1) * sqrt(sum2));
};*/

PackedStringArray Rag::absolute_split_text(String text, int index) {
	if ((index >= absolute_separators.size()) || (text.length() <= chunk_size)) {
		PackedStringArray array{ PackedStringArray() };
		array.append(text);
		return array;
	} else {
		PackedStringArray array{ PackedStringArray() };
		PackedStringArray subarray = text.split(absolute_separators[index], false);
		for (String s : subarray) {
			array.append_array(absolute_split_text(s, index + 1));
		}

		return array;
	}
}
PackedStringArray Rag::chunk_split_text(String text, int index) {
	if (index >= chunk_separators.size()) {
		printf("Failed to split text in chunk\n");
		PackedStringArray array{ PackedStringArray() };
		array.append(text);
		return array;
	} else {
		PackedStringArray array = text.split(chunk_separators[index], false);
		bool is_chunk_fit = true;
		for (String s : array) {
			if (s.length() > chunk_size) {
				is_chunk_fit = false;
			}
		}

		if (!is_chunk_fit) {
			return chunk_split_text(text, index + 1);
		} else {
			PackedStringArray chunk_array{ PackedStringArray() };

			String separator = chunk_separators[index];
			int separator_size = separator.length();

			printf("Using separator: %s\n", separator.utf8().get_data());
			printf("Separator size: %d\n", separator_size);
			printf("array size: %I64d\n", array.size());

			int end_index = 0;
			String s = array[end_index];

			while (end_index < array.size()) {
				end_index += 1;
				while ((end_index < array.size()) && (s.length() + separator_size + array[end_index].length() < chunk_size)) {
					s += separator + array[end_index];
					end_index += 1;
				}

				printf("Chunk: %s\n", s.utf8().get_data());
				printf("Chunk length: %d\n", s.length());
				chunk_array.append(s);

				// Add overlap
				if (end_index < array.size()) {
					s = array[end_index];
					int start_index = end_index - 1;
					while ((start_index >= 0) && (array[start_index].length() + separator_size + s.length() < chunk_overlap)) {
						s = array[start_index] + separator + s;
						start_index -= 1;
					}
					printf("Overlap: %s\n", s.utf8().get_data());
					printf("Overlap length: %d\n", s.length());
				}
			}

			return chunk_array;
		}
	}
}
