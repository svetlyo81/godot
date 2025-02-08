
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

Rag::Rag() {
	if (didInstantiate) {
		printf("There can be only one Rag instance\n");
		return;
	} else didInstantiate = true;

	int rc = SQLITE_OK;
	rc = sqlite3_auto_extension((void (*)())sqlite3_vec_init);
	if (rc != SQLITE_OK) {
		printf("Unable to load sqlite3_vec extension\n");
	}

	params.n_threads = cpu_get_num_math();
	if (params.n_threads > 8) params.n_threads = 8;

	//params.use_mmap = false;
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
	rc = sqlite3_open_v2(dbPath.c_str(), &db, SQLITE_OPEN_READONLY | SQLITE_OPEN_URI, nullptr);
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
	ClassDB::bind_method(D_METHOD("retrieve_similar_texts", "text", "where", "n_results"), &Rag::retrieve_similar_texts);
}

void Rag::set_path(const String &modelPath) {
	params.model = std::string(modelPath.utf8().get_data());
}
void Rag::set_db_path(const String &dbPath_) {
	dbPath = std::string(dbPath_.utf8().get_data());
}

void Rag::initialize() {
	if (model != NULL) {
		printf("Rag is initialized already");
		return;
	}

	openDB();

	tableName = "llm_table";

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
		return;
	}
}

PackedStringArray Rag::retrieve_similar_texts(String text, String where, int n_results) {
	printf("retrieve_similar_texts\n");
	std::vector<float> embedding = compute_embedding(std::string(text.utf8().get_data()));

	String statement_filter = "SELECT rowid FROM " + tableName;
	if (!where.is_empty()) {
		statement_filter += " WHERE " + where;
	}

	String statement_virtual = "SELECT rowid FROM " + tableName + "_virtual WHERE embedding MATCH '[";

	for (float f : embedding) {
		statement_virtual += String::num_real(f) + ", ";
	}

	statement_virtual = statement_virtual.trim_suffix(", ");
	statement_virtual += "]' ";

	statement_virtual += "AND rowid in (" + statement_filter + ") ";

	statement_virtual += "ORDER BY distance LIMIT " + String::num_int64(n_results);

	String statement_final = "SELECT llm_text FROM " + tableName + " WHERE rowid IN (" + statement_virtual + ");";

	//printf("retrieve_text_array statement: %s", statement_final.utf8().get_data());

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

std::vector<float> Rag::compute_embedding(
		std::string prompt
		//, std::function<void(std::vector<float>)> on_compute_finished
) {
	params.prompt = prompt;

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
	//llama_free(ctx);
	//llama_free_model(model);
	//llama_backend_free();

	//on_compute_finished(embeddings);

	return embeddings;
}

float Rag::similarity_cos(std::vector<float> embd1, std::vector<float> embd2) {
	if (embd1.size() != embd2.size()) {
		printf("Error: embedding sizes don't match");
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
