
#ifndef GDLLAMA_RAG_H
#define GDLLAMA_RAG_H

#include "core/object/ref_counted.h"
#include <functional>

class Rag : public RefCounted {
	GDCLASS(Rag, RefCounted);

private:
	std::string dbPath = "";
	String tableName = "";

	void openDB();
	void closeDB();

protected:
	static void _bind_methods();

public:
	void set_path(const String &modelPath);
	void set_db_path(const String &dbPath_);
	void initialize();
	PackedStringArray retrieve_similar_texts(String text, String where, int n_results);
	std::vector<float> compute_embedding(
			std::string prompt
			//, std::function<void(std::vector<float>)> on_compute_finished
	);
	float similarity_cos(std::vector<float> embd1, std::vector<float> embd2);

	Rag();
	~Rag();
};

#endif
