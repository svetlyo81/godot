
#ifndef GDLLAMA_H
#define GDLLAMA_H

#include "core/object/ref_counted.h"
#include "core/os/semaphore.h"

class Llama : public RefCounted {
	GDCLASS(Llama, RefCounted);

private:
	bool isRunning = false;
	bool semaIsWaiting = false;
	bool shouldBreak = false;
	bool shouldSaveSession = false;
	bool shouldLoadSession = false;
	bool didSetSystemPrompt = false;

	bool shouldUseGPU = true;
	int gpuFreeMem = 6000;
	int gpuTotalMem = 0;
	int gpuLayerMem = 245; //gemma9b:200

	std::string buffer = "";
	Semaphore *sema = nullptr;

	//bool isGeneratingXML = false;

	std::string promptPrefixSys = "";
	std::string promptSuffixSys = "";
	std::string promptPrefixSysReply = "";
	std::string promptSuffixSysReply = "";
	std::string promptPrefix = "";
	std::string promptSuffix = "";

protected:
	static void _bind_methods();

public:
	void paramsDefault();

	bool is_running();
	void set_should_use_gpu(bool _shouldUseGPU);
	bool set_gpu_free_mem(int _gpuFreeMem);
	void set_gpu_layer_mem(int _gpuLayerMem);
	void set_path(const String &modelPath);
	void set_session_path(const String &sessionPath);
	void set_grammar(const String &grammarString);
	void set_template(const PackedStringArray &array);
	void set_param(const String &paramName_, float paramValue);
	void set_sys_prompt(const String &promptString_, const String &promptReplyString_);
	void set_prompt(const String &promptString);
	void submit();
	void stop();
	void initialize();
	void start();
	void freeModel();

	Llama();
	~Llama();
};

#endif
