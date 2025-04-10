
#ifndef GDLLAMA_RAG_H
#define GDLLAMA_RAG_H

#include "core/object/ref_counted.h"
#include "core/io/json.h"
#include "core/variant/typed_array.h"
#include <functional>

enum LlmDBMetaDataType {
	INTEGER = 0,
	REAL = 1,
	TEXT = 2,
	BLOB = 3
};

class LlmDBMetaData : public Resource {
	GDCLASS(LlmDBMetaData, Resource)

private:
	String data_name;
	int data_type; //0: integer, 1: real, 2: text, 3: blob

protected:
	static void _bind_methods();

public:
	LlmDBMetaData();
	~LlmDBMetaData();
	static Ref<LlmDBMetaData> create(String data_name, int data_type);
	static Ref<LlmDBMetaData> create_int(String data_name);
	static Ref<LlmDBMetaData> create_real(String data_name);
	static Ref<LlmDBMetaData> create_text(String data_name);
	static Ref<LlmDBMetaData> create_blob(String data_name);
	String get_data_name() const;
	void set_data_name(const String p_data_name);
	int get_data_type() const;
	void set_data_type(const int p_data_type);
};

class Rag : public RefCounted {
	GDCLASS(Rag, RefCounted);

private:
	std::string dbPath = "";

	PackedStringArray absolute_separators;
	PackedStringArray chunk_separators;
	int chunk_size;
	int chunk_overlap;
	int embedding_size;
	TypedArray<LlmDBMetaData> meta;

	void openDB();
	void closeDB();

protected:
	static void _bind_methods();

public:
	void set_path(const String &modelPath);
	void set_db_path(const String &dbPath_);
	void initialize();
	PackedStringArray retrieve_similar_texts(const String text, const String where, const int n_results);
	bool create_db();
	bool insert_text(const String text, const Dictionary meta_dict);
	PackedStringArray split_text(const String text);

	bool execute(String statement);
	std::vector<float> compute_embedding(
			std::string prompt
			//, std::function<void(std::vector<float>)> on_compute_finished
	);
	float similarity_cos(std::vector<float> embd1, std::vector<float> embd2);
	PackedStringArray absolute_split_text(String text, int index);
	PackedStringArray chunk_split_text(String text, int index);

	String type_int_to_string(int meta_data_type);
	Variant::Type type_int_to_variant(int meta_data_type);
	Rag();
	~Rag();
};

#endif
