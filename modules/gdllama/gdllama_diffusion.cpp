
#include "gdllama_diffusion.h"
#include "gdllama_vision.h"

#include <vector>
#include <iostream>
#include <string>
#include <filesystem>
#include <regex>

#include <windows.h>

#include "diffusion-win/stable-diffusion.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "diffusion-win/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "diffusion-win/stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include "diffusion-win/stb_image_resize.h"

namespace fs = std::filesystem;

static void print_utf8(FILE *stream, const char *utf8) {
	if (!utf8) {
		return;
	}

#ifdef _WIN32
	HANDLE h = (stream == stderr)
			? GetStdHandle(STD_ERROR_HANDLE)
			: GetStdHandle(STD_OUTPUT_HANDLE);

	DWORD mode;
	BOOL is_console = GetConsoleMode(h, &mode);

	if (is_console) {
		int wlen = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, NULL, 0);
		if (wlen <= 0) {
			return;
		}

		wchar_t *wbuf = (wchar_t *)malloc(wlen * sizeof(wchar_t));
		if (!wbuf) {
			return;
		}

		MultiByteToWideChar(CP_UTF8, 0, utf8, -1, wbuf, wlen);

		DWORD written;
		WriteConsoleW(h, wbuf, wlen - 1, &written, NULL);

		free(wbuf);
	} else {
		//DWORD written;
		//WriteFile(h, utf8, (DWORD)strlen(utf8), &written, NULL);
	}
#else
	fputs(utf8, stream);
#endif
}

static void log_print(enum sd_log_level_t level, const char *log, bool verbose, bool color) {
	int tag_color;
	const char *level_str;
	FILE *out_stream = (level == SD_LOG_ERROR) ? stderr : stdout;

	if (!log || (!verbose && level <= SD_LOG_DEBUG)) {
		return;
	}

	switch (level) {
		case SD_LOG_DEBUG:
			tag_color = 37;
			level_str = "DEBUG";
			break;
		case SD_LOG_INFO:
			tag_color = 34;
			level_str = "INFO";
			break;
		case SD_LOG_WARN:
			tag_color = 35;
			level_str = "WARN";
			break;
		case SD_LOG_ERROR:
			tag_color = 31;
			level_str = "ERROR";
			break;
		default: /* Potential future-proofing */
			tag_color = 33;
			level_str = "?????";
			break;
	}

	if (color) {
		fprintf(out_stream, "\033[%d;1m[%-5s]\033[0m ", tag_color, level_str);
	} else {
		fprintf(out_stream, "[%-5s] ", level_str);
	}
	print_utf8(out_stream, log);
	fflush(out_stream);
}

static std::string vec_str_to_string(const std::vector<std::string> &v) {
	std::ostringstream oss;
	oss << "[";
	for (size_t i = 0; i < v.size(); i++) {
		oss << "\"" << v[i] << "\"";
		if (i + 1 < v.size()) {
			oss << ", ";
		}
	}
	oss << "]";
	return oss.str();
}

template <typename T>
static std::string vec_to_string(const std::vector<T> &v) {
	std::ostringstream oss;
	oss << "[";
	for (size_t i = 0; i < v.size(); i++) {
		oss << v[i];
		if (i + 1 < v.size()) {
			oss << ", ";
		}
	}
	oss << "]";
	return oss.str();
}

static bool is_absolute_path(const std::string &p) {
#ifdef _WIN32
	// Windows: C:/path or C:\path
	return p.size() > 1 && std::isalpha(static_cast<unsigned char>(p[0])) && p[1] == ':';
#else
	return !p.empty() && p[0] == '/';
#endif
}

enum SDMode {
	IMG_GEN,
	VID_GEN,
	CONVERT,
	UPSCALE,
	MODE_COUNT
};

struct SDCliParams {
	SDMode mode = IMG_GEN;
	std::string output_path = "output.png";
	int output_begin_idx = -1;

	bool verbose = false;
	bool canny_preprocess = false;
	bool convert_name = false;

	preview_t preview_method = PREVIEW_NONE;
	int preview_interval = 1;
	std::string preview_path = "preview.png";
	int preview_fps = 16;
	bool taesd_preview = false;
	bool preview_noisy = false;
	bool color = false;

	bool normal_exit = false;
};

void sd_log_cb(enum sd_log_level_t level, const char *log, void *data) {
	SDCliParams *cli_params = (SDCliParams *)data;
	log_print(level, log, cli_params->verbose, cli_params->color);
}

struct SDContextParams {
	int n_threads = -1;
	std::string model_path;
	std::string clip_l_path;
	std::string clip_g_path;
	std::string clip_vision_path;
	std::string t5xxl_path;
	std::string llm_path;
	std::string llm_vision_path;
	std::string diffusion_model_path;
	std::string high_noise_diffusion_model_path;
	std::string vae_path;
	std::string taesd_path;
	std::string esrgan_path;
	std::string control_net_path;
	std::string embedding_dir;
	std::string photo_maker_path;
	sd_type_t wtype = SD_TYPE_COUNT;
	std::string tensor_type_rules;
	std::string lora_model_dir = ".";

	std::map<std::string, std::string> embedding_map;
	std::vector<sd_embedding_t> embedding_vec;

	rng_type_t rng_type = CUDA_RNG;
	rng_type_t sampler_rng_type = RNG_TYPE_COUNT;
	bool offload_params_to_cpu = false;
	bool enable_mmap = false;
	bool control_net_cpu = false;
	bool clip_on_cpu = false;
	bool vae_on_cpu = false;
	bool diffusion_flash_attn = false;
	bool diffusion_conv_direct = false;
	bool vae_conv_direct = false;

	bool circular = false;
	bool circular_x = false;
	bool circular_y = false;

	bool chroma_use_dit_mask = true;
	bool chroma_use_t5_mask = false;
	int chroma_t5_mask_pad = 1;

	bool qwen_image_zero_cond_t = false;

	prediction_t prediction = PREDICTION_COUNT;
	lora_apply_mode_t lora_apply_mode = LORA_APPLY_AUTO;

	sd_tiling_params_t vae_tiling_params = { false, 0, 0, 0.5f, 0.0f, 0.0f };
	bool force_sdxl_vae_conv_scale = false;

	float flow_shift = INFINITY;

	bool ggml_mxfp4;

	std::string to_string() const {
		std::ostringstream emb_ss;
		emb_ss << "{\n";
		for (auto it = embedding_map.begin(); it != embedding_map.end(); ++it) {
			emb_ss << "    \"" << it->first << "\": \"" << it->second << "\"";
			if (std::next(it) != embedding_map.end()) {
				emb_ss << ",";
			}
			emb_ss << "\n";
		}
		emb_ss << "  }";

		std::string embeddings_str = emb_ss.str();
		std::ostringstream oss;
		oss << "SDContextParams {\n"
			<< "  n_threads: " << n_threads << ",\n"
			<< "  model_path: \"" << model_path << "\",\n"
			<< "  clip_l_path: \"" << clip_l_path << "\",\n"
			<< "  clip_g_path: \"" << clip_g_path << "\",\n"
			<< "  clip_vision_path: \"" << clip_vision_path << "\",\n"
			<< "  t5xxl_path: \"" << t5xxl_path << "\",\n"
			<< "  llm_path: \"" << llm_path << "\",\n"
			<< "  llm_vision_path: \"" << llm_vision_path << "\",\n"
			<< "  diffusion_model_path: \"" << diffusion_model_path << "\",\n"
			<< "  high_noise_diffusion_model_path: \"" << high_noise_diffusion_model_path << "\",\n"
			<< "  vae_path: \"" << vae_path << "\",\n"
			<< "  taesd_path: \"" << taesd_path << "\",\n"
			<< "  esrgan_path: \"" << esrgan_path << "\",\n"
			<< "  control_net_path: \"" << control_net_path << "\",\n"
			<< "  embedding_dir: \"" << embedding_dir << "\",\n"
			<< "  embeddings: " << embeddings_str << "\n"
			<< "  wtype: " << sd_type_name(wtype) << ",\n"
			<< "  tensor_type_rules: \"" << tensor_type_rules << "\",\n"
			<< "  lora_model_dir: \"" << lora_model_dir << "\",\n"
			<< "  photo_maker_path: \"" << photo_maker_path << "\",\n"
			<< "  rng_type: " << sd_rng_type_name(rng_type) << ",\n"
			<< "  sampler_rng_type: " << sd_rng_type_name(sampler_rng_type) << ",\n"
			<< "  flow_shift: " << (std::isinf(flow_shift) ? "INF" : std::to_string(flow_shift)) << "\n"
			<< "  offload_params_to_cpu: " << (offload_params_to_cpu ? "true" : "false") << ",\n"
			<< "  enable_mmap: " << (enable_mmap ? "true" : "false") << ",\n"
			<< "  control_net_cpu: " << (control_net_cpu ? "true" : "false") << ",\n"
			<< "  clip_on_cpu: " << (clip_on_cpu ? "true" : "false") << ",\n"
			<< "  vae_on_cpu: " << (vae_on_cpu ? "true" : "false") << ",\n"
			<< "  diffusion_flash_attn: " << (diffusion_flash_attn ? "true" : "false") << ",\n"
			<< "  diffusion_conv_direct: " << (diffusion_conv_direct ? "true" : "false") << ",\n"
			<< "  vae_conv_direct: " << (vae_conv_direct ? "true" : "false") << ",\n"
			<< "  circular: " << (circular ? "true" : "false") << ",\n"
			<< "  circular_x: " << (circular_x ? "true" : "false") << ",\n"
			<< "  circular_y: " << (circular_y ? "true" : "false") << ",\n"
			<< "  chroma_use_dit_mask: " << (chroma_use_dit_mask ? "true" : "false") << ",\n"
			<< "  qwen_image_zero_cond_t: " << (qwen_image_zero_cond_t ? "true" : "false") << ",\n"
			<< "  chroma_use_t5_mask: " << (chroma_use_t5_mask ? "true" : "false") << ",\n"
			<< "  chroma_t5_mask_pad: " << chroma_t5_mask_pad << ",\n"
			<< "  prediction: " << sd_prediction_name(prediction) << ",\n"
			<< "  lora_apply_mode: " << sd_lora_apply_mode_name(lora_apply_mode) << ",\n"
			<< "  vae_tiling_params: { "
			<< vae_tiling_params.enabled << ", "
			<< vae_tiling_params.tile_size_x << ", "
			<< vae_tiling_params.tile_size_y << ", "
			<< vae_tiling_params.target_overlap << ", "
			<< vae_tiling_params.rel_size_x << ", "
			<< vae_tiling_params.rel_size_y << " },\n"
			<< "  force_sdxl_vae_conv_scale: " << (force_sdxl_vae_conv_scale ? "true" : "false") << "\n"
			<< "}";
		return oss.str();
	}

	sd_ctx_params_t to_sd_ctx_params_t(bool vae_decode_only, bool free_params_immediately, bool taesd_preview) {
		embedding_vec.clear();
		embedding_vec.reserve(embedding_map.size());
		for (const auto &kv : embedding_map) {
			sd_embedding_t item;
			item.name = kv.first.c_str();
			item.path = kv.second.c_str();
			embedding_vec.emplace_back(item);
		}

		sd_ctx_params_t sd_ctx_params = {
			model_path.c_str(),
			clip_l_path.c_str(),
			clip_g_path.c_str(),
			clip_vision_path.c_str(),
			t5xxl_path.c_str(),
			llm_path.c_str(),
			llm_vision_path.c_str(),
			diffusion_model_path.c_str(),
			high_noise_diffusion_model_path.c_str(),
			vae_path.c_str(),
			taesd_path.c_str(),
			control_net_path.c_str(),
			embedding_vec.data(),
			static_cast<uint32_t>(embedding_vec.size()),
			photo_maker_path.c_str(),
			tensor_type_rules.c_str(),
			vae_decode_only,
			free_params_immediately,
			n_threads,
			wtype,
			rng_type,
			sampler_rng_type,
			prediction,
			lora_apply_mode,
			offload_params_to_cpu,
			enable_mmap,
			clip_on_cpu,
			control_net_cpu,
			vae_on_cpu,
			diffusion_flash_attn,
			taesd_preview,
			diffusion_conv_direct,
			vae_conv_direct,
			circular || circular_x,
			circular || circular_y,
			force_sdxl_vae_conv_scale,
			chroma_use_dit_mask,
			chroma_use_t5_mask,
			chroma_t5_mask_pad,
			qwen_image_zero_cond_t,
			flow_shift,
			ggml_mxfp4
		};
		return sd_ctx_params;
	}
};

struct SDGenerationParams {
	std::string prompt;
	std::string prompt_with_lora; // for metadata record only
	std::string negative_prompt;
	int clip_skip = -1; // <= 0 represents unspecified
	int width = -1;
	int height = -1;
	int batch_count = 1;
	std::string input_image_path;
	std::string end_image_path;
	std::string mask_image_path;
	std::string control_image_path;
	std::vector<std::string> ref_image_paths;
	std::string control_video_path;
	bool auto_resize_ref_image = true;
	bool increase_ref_index = false;

	std::vector<int> skip_layers = { 7, 8, 9 };
	sd_sample_params_t sample_params;

	std::vector<int> high_noise_skip_layers = { 7, 8, 9 };
	sd_sample_params_t high_noise_sample_params;

	std::vector<float> custom_sigmas;

	std::string cache_mode;
	std::string cache_option;
	std::string cache_preset;
	std::string scm_mask;
	bool scm_policy_dynamic = true;
	sd_cache_params_t cache_params{};

	float moe_boundary = 0.875f;
	int video_frames = 1;
	int fps = 16;
	float vace_strength = 1.f;

	float strength = 0.75f;
	float control_strength = 0.9f;

	int64_t seed = 42;

	// Photo Maker
	std::string pm_id_images_dir;
	std::string pm_id_embed_path;
	float pm_style_strength = 20.f;

	int upscale_repeats = 1;
	int upscale_tile_size = 128;

	std::map<std::string, float> lora_map;
	std::map<std::string, float> high_noise_lora_map;
	std::vector<sd_lora_t> lora_vec;

	SDGenerationParams() {
		sd_sample_params_init(&sample_params);
		sd_sample_params_init(&high_noise_sample_params);
	}

	bool process_and_check(SDMode mode, const std::string &lora_model_dir) {
		prompt_with_lora = prompt;
		if (width <= 0) {
			printf("error: the width must be greater than 0\n");
			return false;
		}

		if (height <= 0) {
			printf("error: the height must be greater than 0\n");
			return false;
		}

		if (sample_params.sample_steps <= 0) {
			printf("error: the sample_steps must be greater than 0\n");
			return false;
		}

		if (high_noise_sample_params.sample_steps <= 0) {
			high_noise_sample_params.sample_steps = -1;
		}

		if (strength < 0.f || strength > 1.f) {
			printf("error: can only work with strength in [0.0, 1.0]\n");
			return false;
		}

		sd_cache_params_init(&cache_params);

		auto parse_named_params = [&](const std::string &opt_str) -> bool {
			std::stringstream ss(opt_str);
			std::string token;
			while (std::getline(ss, token, ',')) {
				size_t eq_pos = token.find('=');
				if (eq_pos == std::string::npos) {
					printf("error: cache option '%s' missing '=' separator\n", token.c_str());
					return false;
				}
				std::string key = token.substr(0, eq_pos);
				std::string val = token.substr(eq_pos + 1);
				if (key == "threshold") {
					if (cache_mode == "easycache" || cache_mode == "ucache") {
						cache_params.reuse_threshold = std::stof(val);
					} else {
						cache_params.residual_diff_threshold = std::stof(val);
					}
				} else if (key == "start") {
					cache_params.start_percent = std::stof(val);
				} else if (key == "end") {
					cache_params.end_percent = std::stof(val);
				} else if (key == "decay") {
					cache_params.error_decay_rate = std::stof(val);
				} else if (key == "relative") {
					cache_params.use_relative_threshold = (std::stof(val) != 0.0f);
				} else if (key == "reset") {
					cache_params.reset_error_on_compute = (std::stof(val) != 0.0f);
				} else if (key == "Fn" || key == "fn") {
					cache_params.Fn_compute_blocks = std::stoi(val);
				} else if (key == "Bn" || key == "bn") {
					cache_params.Bn_compute_blocks = std::stoi(val);
				} else if (key == "warmup") {
					cache_params.max_warmup_steps = std::stoi(val);
				} else {
					printf("error: unknown cache parameter '%s'\n", key.c_str());
					return false;
				}
			}
			return true;
		};

		if (!cache_mode.empty()) {
			if (cache_mode == "easycache") {
				cache_params.mode = SD_CACHE_EASYCACHE;
				cache_params.reuse_threshold = 0.2f;
				cache_params.start_percent = 0.15f;
				cache_params.end_percent = 0.95f;
				cache_params.error_decay_rate = 1.0f;
				cache_params.use_relative_threshold = true;
				cache_params.reset_error_on_compute = true;
			} else if (cache_mode == "ucache") {
				cache_params.mode = SD_CACHE_UCACHE;
				cache_params.reuse_threshold = 1.0f;
				cache_params.start_percent = 0.15f;
				cache_params.end_percent = 0.95f;
				cache_params.error_decay_rate = 1.0f;
				cache_params.use_relative_threshold = true;
				cache_params.reset_error_on_compute = true;
			} else if (cache_mode == "dbcache") {
				cache_params.mode = SD_CACHE_DBCACHE;
				cache_params.Fn_compute_blocks = 8;
				cache_params.Bn_compute_blocks = 0;
				cache_params.residual_diff_threshold = 0.08f;
				cache_params.max_warmup_steps = 8;
			} else if (cache_mode == "taylorseer") {
				cache_params.mode = SD_CACHE_TAYLORSEER;
				cache_params.Fn_compute_blocks = 8;
				cache_params.Bn_compute_blocks = 0;
				cache_params.residual_diff_threshold = 0.08f;
				cache_params.max_warmup_steps = 8;
			} else if (cache_mode == "cache-dit") {
				cache_params.mode = SD_CACHE_CACHE_DIT;
				cache_params.Fn_compute_blocks = 8;
				cache_params.Bn_compute_blocks = 0;
				cache_params.residual_diff_threshold = 0.08f;
				cache_params.max_warmup_steps = 8;
			}

			if (!cache_option.empty()) {
				if (!parse_named_params(cache_option)) {
					return false;
				}
			}

			if (cache_mode == "easycache" || cache_mode == "ucache") {
				if (cache_params.reuse_threshold < 0.0f) {
					printf("error: cache threshold must be non-negative\n");
					return false;
				}
				if (cache_params.start_percent < 0.0f || cache_params.start_percent >= 1.0f ||
						cache_params.end_percent <= 0.0f || cache_params.end_percent > 1.0f ||
						cache_params.start_percent >= cache_params.end_percent) {
					printf("error: cache start/end percents must satisfy 0.0 <= start < end <= 1.0\n");
					return false;
				}
			}
		}

		if (cache_params.mode == SD_CACHE_DBCACHE ||
				cache_params.mode == SD_CACHE_TAYLORSEER ||
				cache_params.mode == SD_CACHE_CACHE_DIT) {
			if (!scm_mask.empty()) {
				cache_params.scm_mask = scm_mask.c_str();
			}
			cache_params.scm_policy_dynamic = scm_policy_dynamic;
		}

		sample_params.guidance.slg.layers = skip_layers.data();
		sample_params.guidance.slg.layer_count = skip_layers.size();
		sample_params.custom_sigmas = custom_sigmas.data();
		sample_params.custom_sigmas_count = static_cast<int>(custom_sigmas.size());
		high_noise_sample_params.guidance.slg.layers = high_noise_skip_layers.data();
		high_noise_sample_params.guidance.slg.layer_count = high_noise_skip_layers.size();

		if (mode == VID_GEN && video_frames <= 0) {
			return false;
		}

		if (mode == VID_GEN && fps <= 0) {
			return false;
		}

		if (sample_params.shifted_timestep < 0 || sample_params.shifted_timestep > 1000) {
			return false;
		}

		if (upscale_repeats < 1) {
			return false;
		}

		if (upscale_tile_size < 1) {
			return false;
		}

		if (mode == UPSCALE) {
			if (input_image_path.length() == 0) {
				printf("error: upscale mode needs an init image (--init-img)\n");
				return false;
			}
		}

		return true;
	}

	std::string to_string() const {
		char *sample_params_str = sd_sample_params_to_str(&sample_params);
		char *high_noise_sample_params_str = sd_sample_params_to_str(&high_noise_sample_params);

		std::ostringstream lora_ss;
		lora_ss << "{\n";
		for (auto it = lora_map.begin(); it != lora_map.end(); ++it) {
			lora_ss << "    \"" << it->first << "\": \"" << it->second << "\"";
			if (std::next(it) != lora_map.end()) {
				lora_ss << ",";
			}
			lora_ss << "\n";
		}
		lora_ss << "  }";
		std::string loras_str = lora_ss.str();

		lora_ss = std::ostringstream();
		;
		lora_ss << "{\n";
		for (auto it = high_noise_lora_map.begin(); it != high_noise_lora_map.end(); ++it) {
			lora_ss << "    \"" << it->first << "\": \"" << it->second << "\"";
			if (std::next(it) != high_noise_lora_map.end()) {
				lora_ss << ",";
			}
			lora_ss << "\n";
		}
		lora_ss << "  }";
		std::string high_noise_loras_str = lora_ss.str();

		std::ostringstream oss;
		oss << "SDGenerationParams {\n"
			<< "  loras: \"" << loras_str << "\",\n"
			<< "  high_noise_loras: \"" << high_noise_loras_str << "\",\n"
			<< "  prompt: \"" << prompt << "\",\n"
			<< "  negative_prompt: \"" << negative_prompt << "\",\n"
			<< "  clip_skip: " << clip_skip << ",\n"
			<< "  width: " << width << ",\n"
			<< "  height: " << height << ",\n"
			<< "  batch_count: " << batch_count << ",\n"
			<< "  input_image_path: \"" << input_image_path << "\",\n"
			<< "  end_image_path: \"" << end_image_path << "\",\n"
			<< "  mask_image_path: \"" << mask_image_path << "\",\n"
			<< "  control_image_path: \"" << control_image_path << "\",\n"
			<< "  ref_image_paths: " << vec_str_to_string(ref_image_paths) << ",\n"
			<< "  control_video_path: \"" << control_video_path << "\",\n"
			<< "  auto_resize_ref_image: " << (auto_resize_ref_image ? "true" : "false") << ",\n"
			<< "  increase_ref_index: " << (increase_ref_index ? "true" : "false") << ",\n"
			<< "  pm_id_images_dir: \"" << pm_id_images_dir << "\",\n"
			<< "  pm_id_embed_path: \"" << pm_id_embed_path << "\",\n"
			<< "  pm_style_strength: " << pm_style_strength << ",\n"
			<< "  skip_layers: " << vec_to_string(skip_layers) << ",\n"
			<< "  sample_params: " << sample_params_str << ",\n"
			<< "  high_noise_skip_layers: " << vec_to_string(high_noise_skip_layers) << ",\n"
			<< "  high_noise_sample_params: " << high_noise_sample_params_str << ",\n"
			<< "  custom_sigmas: " << vec_to_string(custom_sigmas) << ",\n"
			<< "  cache_mode: \"" << cache_mode << "\",\n"
			<< "  cache_option: \"" << cache_option << "\",\n"
			<< "  cache: "
			<< (cache_params.mode != SD_CACHE_DISABLED ? "enabled" : "disabled")
			<< " (threshold=" << cache_params.reuse_threshold
			<< ", start=" << cache_params.start_percent
			<< ", end=" << cache_params.end_percent << "),\n"
			<< "  moe_boundary: " << moe_boundary << ",\n"
			<< "  video_frames: " << video_frames << ",\n"
			<< "  fps: " << fps << ",\n"
			<< "  vace_strength: " << vace_strength << ",\n"
			<< "  strength: " << strength << ",\n"
			<< "  control_strength: " << control_strength << ",\n"
			<< "  seed: " << seed << ",\n"
			<< "  upscale_repeats: " << upscale_repeats << ",\n"
			<< "  upscale_tile_size: " << upscale_tile_size << ",\n"
			<< "}";
		free(sample_params_str);
		free(high_noise_sample_params_str);
		return oss.str();
	}
};

static SDCliParams cli_params;
static SDContextParams ctx_params;
static SDGenerationParams gen_params;
static sd_ctx_t *sd_ctx;
static sd_image_t input_image = { (uint32_t)gen_params.width, (uint32_t)gen_params.height, 3, nullptr };
static sd_image_t mask_image = { (uint32_t)gen_params.width, (uint32_t)gen_params.height, 1, nullptr };
static std::vector<sd_image_t> ref_images;
static bool did_set_mask_image = false;
static PackedByteArray outputData;

Diffusion::Diffusion() {
	gen_params.width = 768;
	gen_params.height = 512;
	gen_params.mask_image_path = "";
	ctx_params.diffusion_flash_attn = true;
	ctx_params.flow_shift = 3.0f;
	
	sd_set_log_callback(sd_log_cb, &cli_params);
}
Diffusion::~Diffusion() {
    printf("Diffusion dealloc\n");
}

void Diffusion::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_running"), &Diffusion::is_running);
	ClassDB::bind_method(D_METHOD("set_path", "value"), &Diffusion::set_path, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("set_path_qwen"), &Diffusion::set_path_qwen);
	ClassDB::bind_method(D_METHOD("set_path_klein", "value"), &Diffusion::set_path_klein, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("set_param", "value", "value"), &Diffusion::set_param, DEFVAL(""), DEFVAL(1.0f));
	ClassDB::bind_method(D_METHOD("set_prompt", "value"), &Diffusion::set_prompt, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("set_negative_prompt", "value"), &Diffusion::set_negative_prompt, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("set_image", "value"), &Diffusion::set_image, DEFVAL(NULL));
	ClassDB::bind_method(D_METHOD("set_mask_image", "value"), &Diffusion::set_mask_image, DEFVAL(NULL));
	ClassDB::bind_method(D_METHOD("add_image", "value"), &Diffusion::add_image, DEFVAL(NULL));
	ClassDB::bind_method(D_METHOD("get_alloc_fail_count"), &Diffusion::get_alloc_fail_count);
	ClassDB::bind_method(D_METHOD("start"), &Diffusion::start);
	ClassDB::bind_method(D_METHOD("freeModel"), &Diffusion::freeModel);
}

bool Diffusion::is_running() {
	return isRunning;
}
void Diffusion::set_path(const String &modelPath) {
	ctx_params.model_path = std::string(modelPath.utf8().get_data());
	ctx_params.diffusion_model_path = "";
	ctx_params.vae_path = "";
	ctx_params.llm_path = "";
	ctx_params.qwen_image_zero_cond_t = false;
	ctx_params.offload_params_to_cpu = false;
	ctx_params.ggml_mxfp4 = true;
	gen_params.sample_params.sample_method = EULER_A_SAMPLE_METHOD;
	ctx_params.vae_tiling_params.enabled = true;
}
void Diffusion::set_path_qwen() {
	ctx_params.model_path = "";
	ctx_params.diffusion_model_path = "models/qwen-image-edit-2511.gguf";
	ctx_params.vae_path = "models/qwen_image_vae.safetensors";
	ctx_params.llm_path = "models/Qwen2.5-VL-7B-Instruct.Q8_0.gguf";
	ctx_params.qwen_image_zero_cond_t = true;
	ctx_params.offload_params_to_cpu = true;
	ctx_params.ggml_mxfp4 = false;
	gen_params.sample_params.sample_method = EULER_SAMPLE_METHOD;
	ctx_params.vae_tiling_params.enabled = false;
}
void Diffusion::set_path_klein(const String &modelPath) {
	ctx_params.model_path = "";
	ctx_params.diffusion_model_path = std::string(modelPath.utf8().get_data());
	ctx_params.vae_path = "models/flux2-vae.safetensors";
	ctx_params.llm_path = "models/Qwen3-4B-Q8_0.gguf";
	ctx_params.qwen_image_zero_cond_t = false;
	ctx_params.offload_params_to_cpu = true;
	ctx_params.ggml_mxfp4 = false;
	gen_params.sample_params.sample_method = EULER_SAMPLE_METHOD;
	ctx_params.vae_tiling_params.enabled = false;
}
void Diffusion::set_param(const String &paramName_, const float paramValue) {
	std::string paramName = std::string(paramName_.utf8().get_data());

	if (paramName == "seed") {
		int64_t seed = (int64_t)paramValue;
		shouldRandomizeSeed = (seed==-1);
		gen_params.seed = seed;
	}
	else if (paramName == "strength") gen_params.strength = paramValue;
	else if (paramName == "width") gen_params.width = (int)paramValue;
	else if (paramName == "height") gen_params.height = (int)paramValue;
	else if (paramName == "cfg_scale") gen_params.sample_params.guidance.txt_cfg = paramValue;
	else if (paramName == "guidance") gen_params.sample_params.guidance.distilled_guidance = paramValue;
	else if (paramName == "sample_steps") gen_params.sample_params.sample_steps = (int)paramValue;
	else if (paramName == "sample_method") gen_params.sample_params.sample_method = (sample_method_t)paramValue;
	else if (paramName == "enable_dnn_superres") {
		if (paramValue == 1.0f) enableDNNSuperres = true;
		else enableDNNSuperres = false;
	}
}
void Diffusion::set_prompt(const String &promptString) {
	gen_params.prompt = std::string(promptString.utf8().get_data());
}
void Diffusion::set_negative_prompt(const String &promptString) {
	gen_params.negative_prompt = std::string(promptString.utf8().get_data());
}
void Diffusion::set_image(const PackedByteArray &promptImage) {
	outputData.clear();
	if (input_image.data != nullptr) free(input_image.data);	
	input_image.data = (uint8_t *)stbi_load_from_memory(promptImage.ptr(), promptImage.size(), &inputImageWidth, &inputImageHeight, &inputImageChannels, 3);
}
void Diffusion::set_mask_image(const PackedByteArray &maskImage) {
	mask_image.data = (uint8_t *)stbi_load_from_memory(maskImage.ptr(), maskImage.size(), &maskImageWidth, &maskImageHeight, &maskImageChannels, 1);
	did_set_mask_image = true;
}
void Diffusion::add_image(const PackedByteArray &promptImage) {
	uint8_t *image_buffer = (uint8_t *)stbi_load_from_memory(promptImage.ptr(), promptImage.size(), &inputImageWidth, &inputImageHeight, &inputImageChannels, 3);
	gen_params.width = inputImageWidth;
	gen_params.height = inputImageHeight;
	ref_images.push_back({ (uint32_t)inputImageWidth,
			(uint32_t)inputImageHeight,
			(uint32_t)inputImageChannels,
			image_buffer });
}
int Diffusion::get_alloc_fail_count() {
	return allocFailCount;
}

PackedByteArray Diffusion::start() {
	isRunning = true;

	if (shouldRandomizeSeed) {
		srand((int)time(nullptr));
		gen_params.seed = rand();
	}

	gen_params.process_and_check(IMG_GEN, "");

	outputData.clear();

	bool vae_decode_only = true;

	std::vector<sd_image_t> pmid_images;

	sd_image_t *results = nullptr;
	int num_results = 0;

	if (gen_params.sample_params.sample_method == SAMPLE_METHOD_COUNT) {
		gen_params.sample_params.sample_method = sd_get_default_sample_method(sd_ctx);
	}

	if (gen_params.high_noise_sample_params.sample_method == SAMPLE_METHOD_COUNT) {
		gen_params.high_noise_sample_params.sample_method = sd_get_default_sample_method(sd_ctx);
	}

	if (gen_params.sample_params.scheduler == SCHEDULER_COUNT) {
		gen_params.sample_params.scheduler = sd_get_default_scheduler(sd_ctx, gen_params.sample_params.sample_method);
	}
	
	if (input_image.data != nullptr || ref_images.size()>0) {
		vae_decode_only = false;

		if (inputImageChannels < 3) {
			fprintf(stderr, "the number of channels for the input image must be >= 3, but got %d channels\n", inputImageChannels);

			freeInputBuffers();
			isRunning = false;
			return outputData;
		}
		if (inputImageWidth <= 0) {
			fprintf(stderr, "error: the width of image must be greater than 0\n");

			freeInputBuffers();
			isRunning = false;
			return outputData;
		}
		if (inputImageHeight <= 0) {
			fprintf(stderr, "error: the height of image must be greater than 0\n");

			freeInputBuffers();
			isRunning = false;
			return outputData;
		}

		// Resize input image ...
		if (gen_params.height != inputImageHeight || gen_params.width != inputImageWidth) {
			printf("resize input image from %dx%d to %dx%d\n", inputImageWidth, inputImageHeight, gen_params.width, gen_params.height);

			freeInputBuffers();
			isRunning = false;
			return outputData;
		}
	}

	if (sd_ctx == nullptr) {
		sd_ctx_params_t sd_ctx_params = ctx_params.to_sd_ctx_params_t(vae_decode_only, false, cli_params.taesd_preview);
		sd_ctx = new_sd_ctx(&sd_ctx_params);

		if (sd_ctx == nullptr) {
			printf("new_sd_ctx_t failed\n");

			freeInputBuffers();
			isRunning = false;
			return outputData;
		}
	}

	sd_img_gen_params_t img_gen_params = {
		gen_params.lora_vec.data(),
		static_cast<uint32_t>(gen_params.lora_vec.size()),
		gen_params.prompt.c_str(),
		gen_params.negative_prompt.c_str(),
		gen_params.clip_skip,
		input_image,
		ref_images.data(),
		(int)ref_images.size(),
		gen_params.auto_resize_ref_image,
		gen_params.increase_ref_index,
		mask_image,
		gen_params.width,
		gen_params.height,
		gen_params.sample_params,
		gen_params.strength,
		gen_params.seed,
		gen_params.batch_count,
		{ (uint32_t)gen_params.width, (uint32_t)gen_params.height, 3, nullptr },
		gen_params.control_strength,
		{
				pmid_images.data(),
				(int)pmid_images.size(),
				gen_params.pm_id_embed_path.c_str(),
				gen_params.pm_style_strength,
		}, // pm_params
		ctx_params.vae_tiling_params,
		gen_params.cache_params,
	};

	results = generate_image(sd_ctx, &img_gen_params);
	num_results = gen_params.batch_count;

	if (results == nullptr) {
		printf("generate failed\n");

		freeInputBuffers();
		isRunning = false;
		return outputData;
	}

	if (input_image.data != nullptr) {
		/*if (params.mask_path != "") {
			mask_image_buffer = stbi_load(params.mask_path.c_str(), &maskImageWidth, &maskImageHeight, &maskImageChannels, 1);
		}*/
		if (did_set_mask_image) did_set_mask_image = false;
		else {
			mask_image.data = (uint8_t *)malloc(gen_params.width * gen_params.height);
			memset(mask_image.data, 255, gen_params.width * gen_params.height);
			if (mask_image.data == nullptr) {
				printf("malloc mask image failed\n");

				freeInputBuffers();
				isRunning = false;
				return outputData;
			}

			maskImageWidth = inputImageWidth;
			maskImageHeight = inputImageHeight;
		}
	}

	/*size_t last = params.output_path.find_last_of(".");
	std::string dummy_name = last != std::string::npos ? params.output_path.substr(0, last) : params.output_path;
	for (int i = 0; i < params.batch_count; i++) {
		if (results[i].data == NULL) {
			continue;
		}
		std::string final_image_path = i > 0 ? dummy_name + "_" + std::to_string(i + 1) + ".png" : dummy_name + ".png";
		stbi_write_png(final_image_path.c_str(), results[i].width, results[i].height, results[i].channel,
				results[i].data, 0, "");
		printf("save result image to '%s'\n", final_image_path.c_str());
		free(results[i].data);
		results[i].data = NULL;
	}*/
	
	if (results[0].data != nullptr) {
		cv::Mat pngMat(results[0].height, results[0].width, CV_8UC3, results[0].data);
		if (pngMat.empty()) {
			freeInputBuffers();
			isRunning = false;
			return outputData;
		}
		cv::cvtColor(pngMat, pngMat, cv::COLOR_BGR2RGB);

		if(enableDNNSuperres) Vision::cv_dnn_sr().upsample(pngMat, pngMat);

		std::vector<uchar> pngData;
		std::vector<int> compressionParams = { cv::IMWRITE_PNG_COMPRESSION, 1 };
		bool success = cv::imencode(".png", pngMat, pngData, compressionParams);
		const unsigned char *pngArray = pngData.data();

		pngMat.release();
		free(results[0].data);
		results[0].data = nullptr;

		outputData.resize(pngData.size());
		uint8_t *outputDataPtr = outputData.ptrw();
		for (int i = 0; i < outputData.size(); ++i) {
			outputDataPtr[i] = pngArray[i];
		}
	}

	free(results);

	freeInputBuffers();
	isRunning = false;
	return outputData;
}

void Diffusion::freeInputBuffers() {
	free(input_image.data);
	input_image.data = nullptr;

	free(mask_image.data);
	mask_image.data = nullptr;

	for (auto image : ref_images) {
		free(image.data);
		image.data = nullptr;
	}
	ref_images.clear();
}

void Diffusion::freeModel() {
	if (!isRunning && sd_ctx != nullptr) {
		free_sd_ctx(sd_ctx);
		sd_ctx = nullptr;
	}

	outputData.clear();
}
