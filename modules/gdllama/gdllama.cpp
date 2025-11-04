
#include "gdllama.h"
#include "misc.h"

#include "llama-win/common.h"
#include "llama-win/console.h"
#include "llama-win/llama.h"

#include "llama-win/ggml-vulkan.h"
#include "llama-win/ggml-cuda.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <signal.h>
#include <windows.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267) // possible loss of data
#else

typedef enum VkPhysicalDeviceType {
	VK_PHYSICAL_DEVICE_TYPE_OTHER = 0,
	VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1,
	VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2,
	VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU = 3,
	VK_PHYSICAL_DEVICE_TYPE_CPU = 4,
} VkPhysicalDeviceType;

/*
VK_PHYSICAL_DEVICE_TYPE_OTHER - the device does not match any other available types.
VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU - the device is typically one embedded in or tightly coupled with the host.
VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU - the device is typically a separate processor connected to the host via an interlink.
VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU - the device is typically a virtual node in a virtualization environment.
VK_PHYSICAL_DEVICE_TYPE_CPU - the device is typically running on the same processors as the host.
*/

#endif

static bool didInstantiate = false;
static llama_model *model;
static llama_context *ctx;
static llama_context **g_ctx;
static gpt_params params;
static gpt_params *g_params;
static bool is_interacting = false;
static bool need_insert_eot = false;

static bool file_exists(const std::string &path) {
	std::ifstream f(path.c_str());
	return f.good();
}

static bool file_is_empty(const std::string &path) {
	std::ifstream f;
	f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	f.open(path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
	return f.tellg() == 0;
}

static std::string chat_add_and_format(struct llama_model *model_, std::vector<llama_chat_msg> &chat_msgs, std::string role, std::string content) {
	llama_chat_msg new_msg{ role, content };
	auto formatted = llama_chat_format_single(
			model_, g_params->chat_template, chat_msgs, new_msg, role == "user");
	chat_msgs.push_back({ role, content });
	printf("formatted: %s\n", formatted.c_str());
	return formatted;
}

Llama::Llama() {
	if (didInstantiate) {
		printf("There can be only one Llama instance\n");
		return;
	} else didInstantiate = true;

	params.n_threads = cpu_get_num_math();
	if (params.n_threads > 8) params.n_threads = 8;
	// printf("threads: %d\n", params.n_threads);
	params.interactive = true;
	params.n_ctx = 8192;
	params.prompt_cache_all = true;
	//params.verbose_prompt = true;
	//params.use_mlock = true;

	//params.use_mmap = false;

	params.sparams.temp = 0.35f;
	params.sparams.top_k = 0;
	params.sparams.top_p = 1.0f;
	params.sparams.min_p = 0.05;

	g_params = &params;

	sema = memnew(Semaphore);
}
Llama::~Llama() {
    printf("Llama dealloc\n");

	/*if (sema) {
		memdelete(sema);
	}*/
}

void Llama::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_running"), &Llama::is_running);
	ClassDB::bind_method(D_METHOD("set_should_use_gpu", "value"), &Llama::set_should_use_gpu, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("set_gpu_free_mem", "value"), &Llama::set_gpu_free_mem, DEFVAL(6000));
	ClassDB::bind_method(D_METHOD("set_gpu_layer_mem", "value"), &Llama::set_gpu_layer_mem, DEFVAL(200));
	ClassDB::bind_method(D_METHOD("set_path", "value"), &Llama::set_path, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("set_session_path", "value"), &Llama::set_session_path, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("set_grammar", "value"), &Llama::set_grammar, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("set_template", "value"), &Llama::set_template);
	ClassDB::bind_method(D_METHOD("set_param", "value", "value"), &Llama::set_param, DEFVAL(""), DEFVAL(1.0f));
	ClassDB::bind_method(D_METHOD("set_sys_prompt", "value", "value"), &Llama::set_sys_prompt, DEFVAL(""), DEFVAL(""));
	ClassDB::bind_method(D_METHOD("set_prompt", "value"), &Llama::set_prompt, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("submit"), &Llama::submit);
	ClassDB::bind_method(D_METHOD("stop"), &Llama::stop);
	ClassDB::bind_method(D_METHOD("initialize"), &Llama::initialize);
	ClassDB::bind_method(D_METHOD("start"), &Llama::start);
	ClassDB::bind_method(D_METHOD("freeModel"), &Llama::freeModel);
	ADD_SIGNAL(MethodInfo("response", PropertyInfo(Variant::STRING, "name")));
	ADD_SIGNAL(MethodInfo("finish"));

    //ClassDB::bind_method(D_METHOD("add", "value"), &Llama::add);
    //ClassDB::bind_method(D_METHOD("reset"), &Llama::reset);
    //ClassDB::bind_method(D_METHOD("get_total"), &Llama::get_total);
}

bool Llama::is_running() {
	return isRunning;
}
void Llama::set_should_use_gpu(bool _shouldUseGPU) {
	params.n_gpu_layers = -1;
	shouldUseGPU = _shouldUseGPU;
}
bool Llama::set_gpu_free_mem(int _gpuFreeMem) {
	if (gpuTotalMem > _gpuFreeMem - 100) {
		gpuFreeMem = _gpuFreeMem;
		return true;
	} else return false;
}
void Llama::set_gpu_layer_mem(int _gpuLayerMem) {
	gpuLayerMem = _gpuLayerMem;
}
void Llama::set_path(const String &modelPath) {
	params.model = std::string(modelPath.utf8().get_data());
}
void Llama::set_session_path(const String &sessionPath) {
	params.path_prompt_cache = std::string(sessionPath.utf8().get_data());
}
void Llama::set_grammar(const String &grammarString) {
	params.sparams.grammar = std::string(grammarString.utf8().get_data());
}
void Llama::set_template(const PackedStringArray &array) {
	promptPrefixSys = std::string(array[0].utf8().get_data());
	promptSuffixSys = std::string(array[1].utf8().get_data());
	promptPrefixSysReply = std::string(array[2].utf8().get_data());
	promptSuffixSysReply = std::string(array[3].utf8().get_data());
	promptPrefix = std::string(array[4].utf8().get_data());
	promptSuffix = std::string(array[5].utf8().get_data());
}
void Llama::set_param(const String &paramName_, const float paramValue) {
	std::string paramName = std::string(paramName_.utf8().get_data());

	if (paramName == "temp") params.sparams.temp = paramValue;
	else if (paramName == "repeat") params.sparams.penalty_repeat = paramValue;
	else if (paramName == "min_p") params.sparams.min_p = paramValue;

	//printf("%f\n", params.sparams.temp);
}
void Llama::set_sys_prompt(const String &promptString_, const String &promptReplyString_) {
	std::string promptString = std::string(promptString_.utf8().get_data());
	std::string promptReplyString = std::string(promptReplyString_.utf8().get_data());

	if (promptPrefixSys.empty()) {
		promptPrefixSys = "[INST]";
		promptSuffixSys = "</s>";
		promptPrefixSysReply = "";
		promptSuffixSysReply = "";
	}
	
	params.prompt = promptPrefixSys + promptString + (promptReplyString.empty() ? promptSuffixSys : promptPrefixSysReply + promptReplyString + promptSuffixSysReply);

	didSetSystemPrompt = true;
}
void Llama::set_prompt(const String &promptString) {
	if (promptPrefix.empty()) {
		promptPrefix = "[INST]";
		promptSuffix = "[/INST]";
	}

	std::string newPrompt = (didSetSystemPrompt ? params.prompt : promptPrefix) + std::string(promptString.utf8().get_data()) + promptSuffix;

	if (didSetSystemPrompt) didSetSystemPrompt = false;

	if (isRunning)
		buffer = newPrompt;
	else
		params.prompt = newPrompt;
}
void Llama::submit() {
	if (semaIsWaiting)
		sema->post();
}
void Llama::stop() {
	if (isRunning) {
		shouldBreak = true;
		if (semaIsWaiting)
			sema->post();
	}
}

void Llama::initialize() {
	if (model != NULL) {
		printf("Llama is initialized already");
		return;
	}

	if (params.model.empty()) return;
	if (params.logits_all || params.embedding) return;
	
	if (params.n_ctx != 0 && params.n_ctx < 8) {
		printf("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
		params.n_ctx = 8;
	}

	if (params.rope_freq_base != 0.0) {
		printf("%s: warning: changing RoPE frequency base to %g.\n", __func__, params.rope_freq_base);
	}

	if (params.rope_freq_scale != 0.0) {
		printf("%s: warning: scaling RoPE frequency by %g.\n", __func__, params.rope_freq_scale);
	}

	printf("%s: llama backend init\n", __func__);
	llama_backend_init();
	llama_numa_init(params.numa);

	#if !defined(_MSC_VER)

	if (shouldUseGPU) {
		ggml_backend_vk_init(0);
		int deviceIndex = ggml_backend_vk_get_device_index();
		int deviceType = ggml_backend_vk_get_device_type(deviceIndex);
		if (deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
			size_t gpu_total_mem;
			size_t gpu_free_mem;
			ggml_backend_vk_get_device_memory(deviceIndex, &gpu_free_mem, &gpu_total_mem);

			int32_t gpu_layer_mem = (int32_t)gpuLayerMem;
			int32_t gpu_total_mem_int = (int32_t)(gpu_total_mem / (1024 * 1024));
			gpuTotalMem = (int)gpu_total_mem_int;
			if (gpu_total_mem_int > (int32_t)gpuFreeMem - 100) {
				gpu_total_mem_int -= (int32_t)gpuFreeMem;
				if (gpu_total_mem_int > 900) {
					params.n_gpu_layers = gpu_total_mem_int / gpu_layer_mem;
					if (params.n_gpu_layers <= 0) params.n_gpu_layers = -1;
				}
			}
			else
				params.n_gpu_layers = -1;
		}
	}

	#else

	if (shouldUseGPU) {
		ggml_backend_t ggmlBackend = ggml_backend_cuda_init(0);
		int deviceIndex = ggml_backend_cuda_get_device_index();
		size_t gpu_total_mem;
		size_t gpu_free_mem;
		ggml_backend_cuda_get_device_memory(deviceIndex, &gpu_free_mem, &gpu_total_mem);

		int32_t gpu_layer_mem = (int32_t)gpuLayerMem;
		int32_t gpu_total_mem_int = (int32_t)(gpu_total_mem / (1024 * 1024));
		gpuTotalMem = (int)gpu_total_mem_int;
		if (gpu_total_mem_int > (int32_t)gpuFreeMem - 100) {
			gpu_total_mem_int -= (int32_t)gpuFreeMem;
			if (gpu_total_mem_int > 900) {
				params.n_gpu_layers = gpu_total_mem_int / gpu_layer_mem;
				if (params.n_gpu_layers <= 0)
					params.n_gpu_layers = -1;
			}
		} else params.n_gpu_layers = -1;
	}

	#endif

	//printf("%d", params.n_gpu_layers);

	// load the model and apply lora adapter, if any

	llama_init_result llama_init = llama_init_from_gpt_params(params);
	model = llama_init.model;

	if (model == NULL && params.n_gpu_layers != -1) {
		params.n_gpu_layers = -1;
		llama_init = llama_init_from_gpt_params(params);
		model = llama_init.model;
	}

	if (model == NULL) {
		fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
		return;
	}

	ctx = llama_init.context;
	if (ctx == NULL) {
		fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, params.model.c_str());
		llama_free_model(model);
	} else {
		llama_free(ctx);
		ctx = NULL;
	}

	llama_backend_free();
}

void Llama::start() {
	if (params.prompt.empty() || model==NULL)
		return;

	isRunning = true;

	llama_backend_init();

	params.seed = time(NULL);
	printf("%s: seed  = %u\n", __func__, params.seed);
	std::mt19937 rng(params.seed);

	struct llama_context_params lparams = llama_context_params_from_gpt_params(params);
	ctx = llama_new_context_with_model(model, lparams);

	llama_sampling_params &sparams = params.sparams;
	//llama_context *ctx_guidance = NULL;
	std::vector<llama_chat_msg> chat_msgs;
	g_ctx = &ctx;

	/* if (sparams.cfg_scale > 1.f) {
		ctx_guidance = llama_new_context_with_model(model, lparams);
	}*/

    const int n_ctx_train = llama_n_ctx_train(model);
	const int n_ctx = llama_n_ctx(ctx);
    printf("n_ctx: %d\n", n_ctx);

    if (n_ctx > n_ctx_train) {
        printf("%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, n_ctx);
    }

	// print chat template example in conversation mode
	/*if (params.conversation) {
		if (params.enable_chat_template) {
			printf("%s: chat template example: %s\n", __func__, llama_chat_format_example(model, params.chat_template).c_str());
		} else {
			printf("%s: in-suffix/prefix is specified, chat template will be disabled\n", __func__);
		}
	}*/

	// printf("%s: chat template example: %s\n", __func__, llama_chat_format_example(model, params.chat_template).c_str());

    // print system information
    {
        printf("\n");
		printf("%s\n", gpt_params_get_system_info(params).c_str());
    }

    std::string path_session = params.path_prompt_cache;
	std::vector<llama_token> session_tokens;

    if (!path_session.empty() && shouldLoadSession) {
		shouldLoadSession = false;

        printf("%s: attempting to load saved session from '%s'\n", __func__, path_session.c_str());
		
		if (!file_exists(path_session)) {
			printf("%s: session file does not exist, will create.\n", __func__);
		} else if (file_is_empty(path_session)) {
			printf("%s: The session file is empty. A new session will be initialized.\n", __func__);
		} else {
			// The file exists and is not empty
			session_tokens.resize(n_ctx);
			size_t n_token_count_out = 0;
			if (!llama_state_load_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
				printf("%s: error: failed to load session file '%s'\n", __func__, path_session.c_str());
				isRunning = false;
				return;
			}
			session_tokens.resize(n_token_count_out);
			printf("%s: loaded a session with prompt size of %d tokens\n", __func__, (int)session_tokens.size());
		}
    }

	const bool add_bos = llama_add_bos_token(model);
	if (!llama_model_has_encoder(model)) {
		printf("add_bos: failed\n");
	}
	printf("add_bos: %d\n", add_bos);

    std::vector<llama_token> embd_inp;

    {
		auto prompt = (params.conversation && params.enable_chat_template)
				? chat_add_and_format(model, chat_msgs, "system", params.prompt) // format the system prompt in conversation mode
				: params.prompt;
		if (params.interactive_first || session_tokens.empty()) {
			printf("tokenize the prompt\n");
			embd_inp = ::llama_tokenize(ctx, prompt, true, true);
		} else {
			printf("use session tokens\n");
			embd_inp = session_tokens;
		}

		printf("prompt: \"%s\"\n", log_tostr(prompt));
		//printf("tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp).c_str());
	}
	
    // Should not run without any tokens
	if (embd_inp.empty()) {
		if (add_bos) {
			embd_inp.push_back(llama_token_bos(model));
			printf("embd_inp was considered empty and bos was added: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp).c_str());
		} else {
			printf("error: input is empty\n");
			isRunning = false;
			return;
		}
	}

    // Tokenize negative prompt
	std::vector<llama_token> guidance_inp;
	int guidance_offset = 0;
	/*int original_prompt_len = 0;
	if (ctx_guidance) {
		printf("cfg_negative_prompt: \"%s\"\n", log_tostr(sparams.cfg_negative_prompt));

		guidance_inp = ::llama_tokenize(ctx_guidance, sparams.cfg_negative_prompt, true, true);
		printf("guidance_inp tokenized: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_guidance, guidance_inp).c_str());

		std::vector<llama_token> original_inp = ::llama_tokenize(ctx, params.prompt, true, true);
		printf("original_inp tokenized: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, original_inp).c_str());

		original_prompt_len = original_inp.size();
		guidance_offset = (int)guidance_inp.size() - original_prompt_len;
		printf("original_prompt_len: %s", log_tostr(original_prompt_len));
		printf("guidance_offset:     %s", log_tostr(guidance_offset));
	}*/

	if ((int)embd_inp.size() > n_ctx - 4) {
		printf("%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int)embd_inp.size(), n_ctx - 4);
		isRunning = false;
		return;
	}

    // debug message about similarity of saved session, if applicable
	size_t n_matching_session_tokens = 0;
	if (!session_tokens.empty()) {
		for (llama_token id : session_tokens) {
			if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
				break;
			}
			n_matching_session_tokens++;
		}
		if (n_matching_session_tokens == embd_inp.size()) {
			printf("%s: using full prompt from session file\n", __func__);
		} else if (n_matching_session_tokens >= embd_inp.size()) {
			printf("%s: session file has exact match for prompt!\n", __func__);
		} else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
			printf("%s: warning: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
					__func__, n_matching_session_tokens, embd_inp.size());
		} else {
			printf("%s: session file matches %zu / %zu tokens of prompt\n",
					__func__, n_matching_session_tokens, embd_inp.size());
		}

		// remove any "future" tokens that we might have inherited from the previous session
		llama_kv_cache_seq_rm(ctx, -1, n_matching_session_tokens, -1);
	}

	printf(
			"recalculate the cached logits (check): embd_inp.empty() %s, n_matching_session_tokens %zu, embd_inp.size() %zu, session_tokens.size() %zu, embd_inp.size() %zu \n",
			log_tostr(embd_inp.empty()), n_matching_session_tokens, embd_inp.size(), session_tokens.size(), embd_inp.size());

	// if we will use the cache for the full prompt without reaching the end of the cache, force
	// reevaluation of the last token to recalculate the cached logits
	if (!embd_inp.empty() && n_matching_session_tokens == embd_inp.size() && session_tokens.size() > embd_inp.size()) {
		printf("recalculate the cached logits (do): session_tokens.resize( %zu ) \n", embd_inp.size() - 1);

		session_tokens.resize(embd_inp.size() - 1);
	}

	// number of tokens to keep when resetting context
	if (params.n_keep < 0 || params.n_keep > (int)embd_inp.size()) {
		params.n_keep = (int)embd_inp.size();
	} else {
		params.n_keep += add_bos; // always keep the BOS token
	}

	/*if (params.conversation) {
		params.interactive_first = true;
	}*/

	// enable interactive mode if interactive start is specified
	if (params.interactive_first) {
		params.interactive = true;
	}

	/*if (params.verbose_prompt) {
		printf("\n");
		printf("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
		printf("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
		for (int i = 0; i < (int)embd_inp.size(); i++) {
			printf("%6d -> '%s'\n", embd_inp[i], llama_token_to_piece(ctx, embd_inp[i]).c_str());
		}

		if (ctx_guidance) {
			printf("\n");
			printf("%s: negative prompt: '%s'\n", __func__, sparams.cfg_negative_prompt.c_str());
			printf("%s: number of tokens in negative prompt = %zu\n", __func__, guidance_inp.size());
			for (int i = 0; i < (int)guidance_inp.size(); i++) {
				printf("%6d -> '%s'\n", guidance_inp[i], llama_token_to_piece(ctx, guidance_inp[i]).c_str());
			}
		}

		if (params.n_keep > add_bos) {
			printf("%s: static prompt based on n_keep: '", __func__);
			for (int i = 0; i < params.n_keep; i++) {
				printf("%s", llama_token_to_piece(ctx, embd_inp[i]).c_str());
			}
			printf("'\n");
		}
		printf("\n");
	}*/

    if (params.interactive) {
		printf("%s: interactive mode on.\n", __func__);

		if (!params.antiprompt.empty()) {
			for (const auto &antiprompt : params.antiprompt) {
				printf("Reverse prompt: '%s'\n", antiprompt.c_str());
				/*if (params.verbose_prompt) {
					auto tmp = ::llama_tokenize(ctx, antiprompt, false, true);
					for (int i = 0; i < (int)tmp.size(); i++) {
						printf("%6d -> '%s'\n", tmp[i], llama_token_to_piece(ctx, tmp[i]).c_str());
					}
				}*/
			}
		}

		if (params.input_prefix_bos) {
			printf("Input prefix with BOS\n");
		}

		if (!params.input_prefix.empty()) {
			printf("Input prefix: '%s'\n", params.input_prefix.c_str());
			/*if (params.verbose_prompt) {
				auto tmp = ::llama_tokenize(ctx, params.input_prefix, true, true);
				for (int i = 0; i < (int)tmp.size(); i++) {
					printf("%6d -> '%s'\n", tmp[i], llama_token_to_piece(ctx, tmp[i]).c_str());
				}
			}*/
		}

		if (!params.input_suffix.empty()) {
			printf("Input suffix: '%s'\n", params.input_suffix.c_str());
			/*if (params.verbose_prompt) {
				auto tmp = ::llama_tokenize(ctx, params.input_suffix, false, true);
				for (int i = 0; i < (int)tmp.size(); i++) {
					printf("%6d -> '%s'\n", tmp[i], llama_token_to_piece(ctx, tmp[i]).c_str());
				}
			}*/
		}
	}
	printf("sampling: \n%s\n", llama_sampling_print(sparams).c_str());
	printf("sampling order: \n%s\n", llama_sampling_order_print(sparams).c_str());
	printf("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);

	// group-attention state
	// number of grouped KV tokens so far (used only if params.grp_attn_n > 1)
	int ga_i = 0;

	const int ga_n = params.grp_attn_n;
	const int ga_w = params.grp_attn_w;

	if (ga_n != 1) {
		//GGML_ASSERT(ga_n > 0 && "grp_attn_n must be positive"); // NOLINT
		//GGML_ASSERT(ga_w % ga_n == 0 && "grp_attn_w must be a multiple of grp_attn_n"); // NOLINT
		//GGML_ASSERT(n_ctx_train % ga_w == 0     && "n_ctx_train must be a multiple of grp_attn_w");    // NOLINT
		//GGML_ASSERT(n_ctx >= n_ctx_train * ga_n && "n_ctx must be at least n_ctx_train * grp_attn_n"); // NOLINT
		printf("self-extend: n_ctx_train = %d, grp_attn_n = %d, grp_attn_w = %d\n", n_ctx_train, ga_n, ga_w);
	}
	printf("\n\n");

	if (params.interactive) {
		/* const char *control_message;
		if (params.multiline_input) {
			control_message = " - To return control to the AI, end your input with '\\'.\n"
							  " - To return control without starting a new line, end your input with '/'.\n";
		} else {
			control_message = " - Press Return to return control to the AI.\n"
							  " - To return control without starting a new line, end your input with '/'.\n"
							  " - If you want to submit another line, end your input with '\\'.\n";
		}
		printf("== Running in interactive mode. ==\n");
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__)) || defined(_WIN32)
		printf(" - Press Ctrl+C to interject at any time.\n");
#endif
		printf("%s\n", control_message);*/

		is_interacting = params.interactive_first;
	}

    bool is_antiprompt = false;
	bool input_echo = true;
	bool is_response = false;
	//bool display = true;
	//bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();
	
    int n_past = 0;
	int n_remain = params.n_predict;
	int n_consumed = 0;
	int n_session_consumed = 0;
	int n_past_guidance = 0;

    //std::vector<int>   input_tokens;
    //std::vector<int>   output_tokens;
    std::ostringstream output_ss;
	//std::ostringstream assistant_ss; // for storing current assistant message, used in conversation mode

	output_ss << params.prompt;

    // the first thing we will do is to output the prompt, so set color accordingly
    //console::set_display(console::prompt);
	//display = params.display_prompt;

    std::vector<llama_token> embd;
    std::vector<llama_token> embd_guidance;
	// tokenized antiprompts
	std::vector<std::vector<llama_token>> antiprompt_ids;

	std::string generate_text_buffer = "";

    antiprompt_ids.reserve(params.antiprompt.size());
	for (const std::string &antiprompt : params.antiprompt) {
		antiprompt_ids.emplace_back(::llama_tokenize(ctx, antiprompt, false, true));
	}

    struct llama_sampling_context *ctx_sampling = llama_sampling_init(sparams);
	if (!ctx_sampling) {
		printf("%s: failed to initialize sampling subsystem\n", __func__);
		isRunning = false;
		return;
	}
	
    if (llama_model_has_encoder(model)) {
		int enc_input_size = embd_inp.size();
		llama_token *enc_input_buf = embd_inp.data();

		if (llama_encode(ctx, llama_batch_get_one(enc_input_buf, enc_input_size, 0, 0))) {
			printf("%s : failed to eval\n", __func__);
			isRunning = false;
			return;
		}

		llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
		if (decoder_start_token_id == -1) {
			decoder_start_token_id = llama_token_bos(model);
		}

		embd_inp.clear();
		embd_inp.push_back(decoder_start_token_id);
	}

    while ((n_remain != 0 && !is_antiprompt) || params.interactive) {
        // predict
        if (!embd.empty()) {
			// Note: (n_ctx - 4) here is to match the logic for commandline prompt handling via
			// --prompt or --file which uses the same value.
			int max_embd_size = n_ctx - 4;

			// Ensure the input doesn't exceed the context size by truncating embd if necessary.
			if ((int)embd.size() > max_embd_size) {
				const int skipped_tokens = (int)embd.size() - max_embd_size;
				embd.resize(max_embd_size);

				//console::set_display(console::error);
				printf("<<input too long: skipped %d token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
				//console::set_display(console::reset);
				fflush(stdout);
			}

            if (ga_n == 1) {
				// infinite text generation via context shifting
				// if we run out of context:
				// - take the n_keep first tokens from the original prompt (via n_past)
				// - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
				if (n_past + (int)embd.size() + std::max<int>(0, guidance_offset) >= n_ctx) {
					if (params.n_predict == -2) {
						printf("\n\n%s: context full and n_predict == -%d => stopping\n", __func__, params.n_predict);
						break;
					}

					const int n_left = n_past - params.n_keep;
					const int n_discard = n_left / 2;

					printf("context full, swapping: n_past = %d, n_left = %d, n_ctx = %d, n_keep = %d, n_discard = %d\n",
							n_past, n_left, n_ctx, params.n_keep, n_discard);

					llama_kv_cache_seq_rm(ctx, 0, params.n_keep, params.n_keep + n_discard);
					llama_kv_cache_seq_add(ctx, 0, params.n_keep + n_discard, n_past, -n_discard);

					n_past -= n_discard;

					/*if (ctx_guidance) {
						n_past_guidance -= n_discard;
					}*/

					printf("after swap: n_past = %d, n_past_guidance = %d\n", n_past, n_past_guidance);

					//printf("embd: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd).c_str());

					printf("clear session path\n");
					//path_session.clear();
				}
			} else {
				// context extension via Self-Extend
				while (n_past >= ga_i + ga_w) {
					const int ib = (ga_n * ga_i) / ga_w;
					const int bd = (ga_w / ga_n) * (ga_n - 1);
					const int dd = (ga_w / ga_n) - ib * bd - ga_w;

					printf("\n");
					printf("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i, n_past, ib * bd, ga_i + ib * bd, n_past + ib * bd);
					printf("div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", ga_i + ib * bd, ga_i + ib * bd + ga_w, ga_n, (ga_i + ib * bd) / ga_n, (ga_i + ib * bd + ga_w) / ga_n);
					printf("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i + ib * bd + ga_w, n_past + ib * bd, dd, ga_i + ib * bd + ga_w + dd, n_past + ib * bd + dd);

					llama_kv_cache_seq_add(ctx, 0, ga_i, n_past, ib * bd);
					llama_kv_cache_seq_div(ctx, 0, ga_i + ib * bd, ga_i + ib * bd + ga_w, ga_n);
					llama_kv_cache_seq_add(ctx, 0, ga_i + ib * bd + ga_w, n_past + ib * bd, dd);

					n_past -= bd;

					ga_i += ga_w / ga_n;

					printf("\nn_past_old = %d, n_past = %d, ga_i = %d\n\n", n_past + bd, n_past, ga_i);
				}
			}

            // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
			if (n_session_consumed < (int)session_tokens.size()) {
				size_t i = 0;
				for (; i < embd.size(); i++) {
					if (embd[i] != session_tokens[n_session_consumed]) {
						session_tokens.resize(n_session_consumed);
						break;
					}

					n_past++;
					n_session_consumed++;

					if (n_session_consumed >= (int)session_tokens.size()) {
						++i;
						break;
					}
				}
				if (i > 0) {
					embd.erase(embd.begin(), embd.begin() + i);
				}
			}

            // evaluate tokens in batches
			// embd is typically prepared beforehand to fit within a batch, but not always
			/*if (ctx_guidance) {
				int input_size = 0;
				llama_token *input_buf = NULL;

				if (n_past_guidance < (int)guidance_inp.size()) {
					// Guidance context should have the same data with these modifications:
					//
					// * Replace the initial prompt
					// * Shift everything by guidance_offset
					embd_guidance = guidance_inp;
					if (embd.begin() + original_prompt_len < embd.end()) {
						embd_guidance.insert(
								embd_guidance.end(),
								embd.begin() + original_prompt_len,
								embd.end());
					}

					input_buf = embd_guidance.data();
					input_size = embd_guidance.size();

					printf("guidance context: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_guidance).c_str());
				} else {
					input_buf = embd.data();
					input_size = embd.size();
				}

				for (int i = 0; i < input_size; i += params.n_batch) {
					int n_eval = std::min(input_size - i, params.n_batch);
					if (llama_decode(ctx_guidance, llama_batch_get_one(input_buf + i, n_eval, n_past_guidance, 0))) {
						printf("%s : failed to eval\n", __func__);
						isRunning = false;
						return;
					}

					n_past_guidance += n_eval;
				}
			}*/

			for (int i = 0; i < (int)embd.size(); i += params.n_batch) {
				int n_eval = (int)embd.size() - i;
				if (n_eval > params.n_batch) {
					n_eval = params.n_batch;
				}

				//printf("eval: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd).c_str());

				if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval, n_past, 0))) {
					printf("%s : failed to eval\n", __func__);
					isRunning = false;
					return;
				}

				n_past += n_eval;

				//printf("n_past = %d\n", n_past);
				// Display total tokens alongside total time
				if (params.n_print > 0 && n_past % params.n_print == 0) {
					printf("\n\033[31mTokens consumed so far = %d / %d \033[0m\n", n_past, n_ctx);
				}
			}

			if (!embd.empty() && !path_session.empty()) {
				session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
				n_session_consumed = session_tokens.size();
			}
        }

        embd.clear();
		embd_guidance.clear();

        if ((int)embd_inp.size() <= n_consumed && !is_interacting) {
			// optionally save the session on first sample (for faster prompt loading next time)
			/*if (!path_session.empty() && shouldSaveSession && !params.prompt_cache_ro) {
				shouldSaveSession = false;
				llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());

				printf("saved session to %s\n", path_session.c_str());
			}*/

			const llama_token id = llama_sampling_sample(ctx_sampling, ctx, NULL/*ctx_guidance*/);

			llama_sampling_accept(ctx_sampling, ctx, id, /* apply_grammar= */ true);

			//printf("last: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, ctx_sampling->prev).c_str());

			embd.push_back(id);

			// echo this to console
			input_echo = true;

			// decrement remaining sampling budget
			--n_remain;

			//printf("n_remain: %d\n", n_remain);
		} else {
			// some user input remains from prompt or interaction, forward it to processing
			printf("embd_inp.size(): %d, n_consumed: %d\n", (int)embd_inp.size(), n_consumed);
			while ((int)embd_inp.size() > n_consumed) {
				embd.push_back(embd_inp[n_consumed]);

				// push the prompt in the sampling context in order to apply repetition penalties later
				// for the prompt, we don't apply grammar rules
				llama_sampling_accept(ctx_sampling, ctx, embd_inp[n_consumed], /* apply_grammar= */ false);

				++n_consumed;
				if ((int)embd.size() >= params.n_batch) {
					break;
				}
			}
		}

		// reset color to default if there is no pending user input
		if (input_echo && (int)embd_inp.size() == n_consumed) {
			//console::set_display(console::reset);
			//display = true;
			is_response = true;
		}

		// display text
		if (input_echo) {
			for (auto id : embd) {
				const std::string token_str = llama_token_to_piece(ctx, id, params.special);

				// Console/Stream Output
				//printf("%s", token_str.c_str());

				// Record Displayed Tokens To Log
				// Note: Generated tokens are created one by one hence this check
				if (embd.size() > 1) {
					// Incoming Requested Tokens
					//input_tokens.push_back(id);

					//if (is_response) emit_signal(SNAME("response"), String(token_str.c_str()));
				} else {
					// Outgoing Generated Tokens
					//output_tokens.push_back(id);
					output_ss << token_str;

					/*bool shouldEndXML = false;
					if (token_str.find(">") != std::string::npos) {
						shouldEndXML = true;
						isGeneratingXML = true;
					}
					else if (token_str.find("<") != std::string::npos)
						isGeneratingXML = true;*/

					if (is_response /*&& !isGeneratingXML*/) {
						if (generate_text_buffer.empty() && is_utf8(token_str.data())) {
							String new_text = string_std_to_gd(token_str);
							emit_signal(SNAME("response"), new_text);
						} else {
							generate_text_buffer.append(token_str);
							if (is_utf8(generate_text_buffer.data())) {
								String new_text = string_std_to_gd(generate_text_buffer);
								generate_text_buffer.clear();
								emit_signal(SNAME("response"), new_text);
							}
						}
					}

					//if (shouldEndXML) isGeneratingXML = false;
				}

				fflush(stdout);
			}
		}

        // if not currently processing queued inputs;
		if ((int)embd_inp.size() <= n_consumed) {
			// check for reverse prompt in the last n_prev tokens
			if (!params.antiprompt.empty()) {
				const int n_prev = 32;
				const std::string last_output = llama_sampling_prev_str(ctx_sampling, ctx, n_prev);

				is_antiprompt = false;
				// Check if each of the reverse prompts appears at the end of the output.
				// If we're not running interactively, the reverse prompt might be tokenized with some following characters
				// so we'll compensate for that by widening the search window a bit.
				for (std::string &antiprompt : params.antiprompt) {
					size_t extra_padding = params.interactive ? 0 : 2;
					size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
							? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
							: 0;

					if (last_output.find(antiprompt, search_start_pos) != std::string::npos) {
						if (params.interactive) {
							is_interacting = true;
						}
						is_antiprompt = true;
						break;
					}
				}

				// check for reverse prompt using special tokens
				llama_token last_token = llama_sampling_last(ctx_sampling);
				for (std::vector<llama_token> ids : antiprompt_ids) {
					if (ids.size() == 1 && last_token == ids[0]) {
						if (params.interactive) {
							is_interacting = true;
						}
						is_antiprompt = true;
						break;
					}
				}

				if (is_antiprompt) {
					printf("found antiprompt: %s\n", last_output.c_str());
				}
			}

            // deal with end of generation tokens in interactive mode
			if (llama_token_is_eog(model, llama_sampling_last(ctx_sampling))) {
				printf("found an EOG token\n");

				if (params.interactive) {
					if (!params.antiprompt.empty()) {
						// tokenize and inject first reverse prompt
						const auto first_antiprompt = ::llama_tokenize(ctx, params.antiprompt.front(), false, true);
						embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
						is_antiprompt = true;
					}

					if (params.enable_chat_template) {
						chat_add_and_format(model, chat_msgs, "assistant", "" /*assistant_ss.str()*/);
					}
					is_interacting = true;
					printf("\n");
				}
			}

			// if current token is not EOG, we add it to current assistant message
			/*if (params.conversation) {
				auto id = llama_sampling_last(ctx_sampling);
				assistant_ss << llama_token_to_piece(ctx, id, false);
			}*/

            if (n_past > 0 && is_interacting) {
                printf("waiting for user input\n");

                /*if (params.conversation) {
					printf("\n> ");
				}*/

                if (params.input_prefix_bos) {
					printf("adding input prefix BOS token\n");
					embd_inp.push_back(llama_token_bos(model));
				}

                buffer = "";
				if (!params.input_prefix.empty() && !params.conversation) {
					printf("appending input prefix: '%s'\n", params.input_prefix.c_str());
					printf("%s", params.input_prefix.c_str());
				}

                // color user input only
				//console::set_display(console::user_input);
				//display = params.display_prompt;

                /*std::string line;
				bool another_line = true;
				do {
					another_line = console::readline(line, params.multiline_input);
					buffer += line;
				} while (another_line);*/

                // done taking input, reset color
				//console::set_display(console::reset);
				//display = true;

				emit_signal(SNAME("finish"));

				if (!shouldBreak) {
					semaIsWaiting = true;
					sema->wait();
				} else {
					semaIsWaiting = false;
					break;
				}

                // Add tokens to embd only if the input buffer is non-empty
				// Entering a empty line lets the user pass control back
				if (buffer.length() > 1) {
					// append input suffix if any
					if (!params.input_suffix.empty() && !params.conversation) {
						printf("appending input suffix: '%s'\n", params.input_suffix.c_str());
						printf("%s", params.input_suffix.c_str());
					}

					printf("buffer: '%s'\n", buffer.c_str());

					const size_t original_size = embd_inp.size();

					if (params.escape) {
						string_process_escapes(buffer);
					}

					bool format_chat = params.conversation && params.enable_chat_template;
					std::string user_inp = format_chat
							? chat_add_and_format(model, chat_msgs, "user", std::move(buffer))
							: std::move(buffer);
					// TODO: one inconvenient of current chat template implementation is that we can't distinguish between user input and special tokens (prefix/postfix)
					const auto line_pfx = ::llama_tokenize(ctx, params.input_prefix, false, true);
					const auto line_inp = ::llama_tokenize(ctx, user_inp, false, format_chat);
					const auto line_sfx = ::llama_tokenize(ctx, params.input_suffix, false, true);

					//printf("input tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, line_inp).c_str());

					// if user stop generation mid-way, we must add EOT to finish model's last response
					if (need_insert_eot && format_chat) {
						llama_token eot = llama_token_eot(model);
						embd_inp.push_back(eot == -1 ? llama_token_eos(model) : eot);
						need_insert_eot = false;
					}

					embd_inp.insert(embd_inp.end(), line_pfx.begin(), line_pfx.end());
					embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
					embd_inp.insert(embd_inp.end(), line_sfx.begin(), line_sfx.end());

					for (size_t i = original_size; i < embd_inp.size(); ++i) {
						const llama_token token = embd_inp[i];
						//output_tokens.push_back(token);
						output_ss << llama_token_to_piece(ctx, token);
					}

					// reset assistant message
					//assistant_ss.str("");

					n_remain -= line_inp.size();
					printf("n_remain: %d\n", n_remain);
				} else {
					shouldBreak = true;
					printf("empty line, passing control back\n");
				}

				input_echo = false; // do not echo this again
            }

            if (n_past > 0) {
				if (is_interacting) {
					llama_sampling_reset(ctx_sampling);
				}
				is_interacting = false;
			}
        }

        // end of generation
		if (!embd.empty() && llama_token_is_eog(model, embd.back()) && !(params.interactive)) {
			printf(" [end of text]\n");
			break;
		}

		// In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
		// We skip this logic when n_predict == -1 (infinite) or -2 (stop at context size).
		if (params.interactive && n_remain <= 0 && params.n_predict >= 0) {
			n_remain = params.n_predict;
			is_interacting = true;
		}

		if (shouldBreak)
			break;
    }
	
	if (!path_session.empty() && params.prompt_cache_all && !params.prompt_cache_ro && shouldSaveSession) {
		shouldSaveSession = false;
		
		printf("\n%s: saving final output to session file '%s'\n", __func__, path_session.c_str());
		llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
	}

    llama_print_timings(ctx);

	llama_kv_cache_clear(ctx);
	llama_reset_timings(ctx);

	//if (ctx_guidance) llama_free(ctx_guidance);
	if (ctx) {
		llama_free(ctx);
		ctx = NULL;
	}

	llama_sampling_free(ctx_sampling);
	llama_backend_free();

	buffer = "";

	isRunning = false;
	shouldBreak = false;
	semaIsWaiting = false;
	//isGeneratingXML = false;

	printf("output_ss: %s\n", output_ss.str().c_str());

	emit_signal(SNAME("finish"));
}

void Llama::freeModel() {
	if (!isRunning && model != NULL) {
		llama_free_model(model);
		model = NULL;
	}
}
