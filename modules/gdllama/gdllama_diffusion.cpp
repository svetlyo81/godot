
#include "gdllama_diffusion.h"

#include <vector>
#include <iostream>
#include <string>

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

enum SDMode {
	TXT2IMG,
	IMG2IMG,
	IMG2VID,
	IMG_GEN,
	VID_GEN,
	CONVERT,
	MODE_COUNT
};

struct SDParams {
	int n_threads = -1;
	SDMode mode = IMG_GEN;
	std::string model_path;
	std::string clip_l_path;
	std::string clip_g_path;
	std::string t5xxl_path;
	std::string diffusion_model_path;
	std::string vae_path;
	std::string taesd_path;
	std::string esrgan_path;
	std::string control_net_path;
	std::string embedding_dir;
	std::string stacked_id_embed_dir;
	std::string input_id_images_path;
	sd_type_t wtype = SD_TYPE_COUNT;
	std::string tensor_type_rules;
	std::string lora_model_dir;
	std::string output_path = "output.png";
	std::string input_path;
	std::string mask_path;
	std::string control_image_path;
	std::vector<std::string> ref_image_paths;

	std::string prompt;
	std::string negative_prompt;
	std::unordered_map<std::string, float> lora_state;
	float min_cfg = 1.0f;
	float cfg_scale = 7.0f;
	float img_cfg_scale = INFINITY;
	float guidance = 3.5f;
	float eta = 0.f;
	float style_ratio = 20.f;
	int clip_skip = -1; // <= 0 represents unspecified
	int width = 512;
	int height = 512;
	int batch_count = 1;

	int video_frames = 6;
	int motion_bucket_id = 127;
	int fps = 6;
	float augmentation_level = 0.f;

	sample_method_t sample_method = EULER_A;
	schedule_t schedule = DEFAULT;
	int sample_steps = 20;
	float strength = 0.75f;
	float control_strength = 0.9f;
	rng_type_t rng_type = CUDA_RNG;
	int64_t seed = 42;
	bool verbose = false;
	bool vae_tiling = false;
	bool control_net_cpu = false;
	bool normalize_input = false;
	bool clip_on_cpu = false;
	bool vae_on_cpu = false;
	bool diffusion_flash_attn = false;
	bool diffusion_conv_direct = false;
	bool vae_conv_direct = false;
	bool canny_preprocess = false;
	bool color = false;
	int upscale_repeats = 1;

	std::vector<int> skip_layers = { 7, 8, 9 };
	float slg_scale = 0.f;
	float skip_layer_start = 0.01f;
	float skip_layer_end = 0.2f;

	bool chroma_use_dit_mask = true;
	bool chroma_use_t5_mask = false;
	int chroma_t5_mask_pad = 1;
};

/* Enables Printing the log level tag in color using ANSI escape codes */
void sd_log_cb(enum sd_log_level_t level, const char *log, void *data) {
	SDParams *params = (SDParams *)data;
	int tag_color;
	const char *level_str;
	FILE *out_stream = (level == SD_LOG_ERROR) ? stderr : stdout;

	if (!log || (!params->verbose && level <= SD_LOG_DEBUG)) {
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

	if (params->color == true) {
		fprintf(out_stream, "\033[%d;1m[%-5s]\033[0m ", tag_color, level_str);
	} else {
		fprintf(out_stream, "[%-5s] ", level_str);
	}
	fputs(log, out_stream);
	fflush(out_stream);
}

static bool didInstantiate = false;
static SDParams params;
static sd_ctx_t *sd_ctx;
static uint8_t *input_image_buffer = NULL;
static uint8_t *control_image_buffer = NULL;
static uint8_t *mask_image_buffer = NULL;
static bool did_set_mask_image = false;
std::vector<sd_image_t> ref_images;
static PackedByteArray outputData;

Diffusion::Diffusion() {
	if (didInstantiate) {
		printf("There can be only one Diffusion instance\n");
		return;
	} else didInstantiate = true;

	params.n_threads = get_num_physical_cores();
	params.width = 768;
	params.height = 512;
	params.seed = -1;
	params.mask_path = "";

	sd_set_log_callback(sd_log_cb, (void *)&params);
}
Diffusion::~Diffusion() {
    printf("Diffusion dealloc\n");
}

void Diffusion::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_running"), &Diffusion::is_running);
	ClassDB::bind_method(D_METHOD("set_path", "value"), &Diffusion::set_path, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("set_control_path", "value"), &Diffusion::set_control_path, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("set_param", "value", "value"), &Diffusion::set_param, DEFVAL(""), DEFVAL(1.0f));
	ClassDB::bind_method(D_METHOD("set_prompt", "value"), &Diffusion::set_prompt, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("set_negative_prompt", "value"), &Diffusion::set_negative_prompt, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("set_lora_state"), &Diffusion::set_lora_state, DEFVAL(NULL));
	ClassDB::bind_method(D_METHOD("set_image", "value"), &Diffusion::set_image, DEFVAL(NULL));
	ClassDB::bind_method(D_METHOD("set_control_image", "value"), &Diffusion::set_control_image, DEFVAL(NULL));
	ClassDB::bind_method(D_METHOD("set_mask_image", "value"), &Diffusion::set_mask_image, DEFVAL(NULL));
	ClassDB::bind_method(D_METHOD("get_alloc_fail_count"), &Diffusion::get_alloc_fail_count);
	ClassDB::bind_method(D_METHOD("start"), &Diffusion::start);
	ClassDB::bind_method(D_METHOD("freeModel"), &Diffusion::freeModel);
}

bool Diffusion::is_running() {
	return isRunning;
}
void Diffusion::set_path(const String &modelPath) {
	params.model_path = std::string(modelPath.utf8().get_data());
}
void Diffusion::set_control_path(const String &controlPath) {
	params.control_net_path = std::string(controlPath.utf8().get_data());
}
void Diffusion::set_param(const String &paramName_, const float paramValue) {
	std::string paramName = std::string(paramName_.utf8().get_data());

	if (paramName == "seed") params.seed = (int64_t)paramValue;
	else if (paramName == "strength") params.strength = paramValue;
	else if (paramName == "control_strength") params.control_strength = paramValue;
	else if (paramName == "width") params.width = (int)paramValue;
	else if (paramName == "height") params.height = (int)paramValue;
	else if (paramName == "cfg_scale") params.cfg_scale = paramValue;
	else if (paramName == "guidance") params.guidance = paramValue;
	else if (paramName == "sample_steps") params.sample_steps = (int)paramValue;
	else if (paramName == "sample_method") params.sample_method = (sample_method_t)paramValue;
	else if (paramName == "enable_taesd") {
		if (paramValue == 1.0f) params.taesd_path = "taesd.safetensors";
		else params.taesd_path = "";
	}
	else if (paramName == "enable_vae_tiling") {
		if (paramValue == 1.0f) params.vae_tiling = true;
		else params.vae_tiling = false;
	}
}
void Diffusion::set_prompt(const String &promptString) {
	params.prompt = std::string(promptString.utf8().get_data());
}
void Diffusion::set_negative_prompt(const String &promptString) {
	params.negative_prompt = std::string(promptString.utf8().get_data());
}
void Diffusion::set_image(const PackedByteArray &promptImage) {
	outputData.clear();
	if (input_image_buffer != NULL) free(input_image_buffer);

	unsigned char *imageData = new unsigned char[promptImage.size()];
	for (int i = 0; i < promptImage.size(); ++i) {
		imageData[i] = promptImage[i];
	}

	input_image_buffer = stbi_load_from_memory(imageData, promptImage.size(), &inputImageWidth, &inputImageHeight, &inputImageChannels, 3);
	delete[] imageData;
}
void Diffusion::set_control_image(const PackedByteArray &promptImage) {
	if (control_image_buffer != NULL) free(control_image_buffer);

	unsigned char *imageData = new unsigned char[promptImage.size()];
	for (int i = 0; i < promptImage.size(); ++i) {
		imageData[i] = promptImage[i];
	}

	control_image_buffer = stbi_load_from_memory(imageData, promptImage.size(), &controlImageWidth, &controlImageHeight, &controlImageChannels, 3);
	delete[] imageData;
}
void Diffusion::set_mask_image(const PackedByteArray &maskImage) {
	unsigned char *imageData = new unsigned char[maskImage.size()];
	for (int i = 0; i < maskImage.size(); ++i) {
		imageData[i] = maskImage[i];
	}

	mask_image_buffer = stbi_load_from_memory(imageData, maskImage.size(), &maskImageWidth, &maskImageHeight, &maskImageChannels, 1);
	delete[] imageData;
	did_set_mask_image = true;
}
int Diffusion::get_alloc_fail_count() {
	return allocFailCount;
}
void Diffusion::set_lora_state(const Dictionary &loraDict) {
	Array keys = loraDict.keys();
	Array values = loraDict.values();
	std::unordered_map<std::string, float> filename2multiplier;

	for (int i = 0; i < loraDict.size(); ++i) {
		String key = (String)keys[i];
		std::string filename = std::string(key.utf8().get_data());
		float multiplier = (float)values[i];

		if (multiplier == 0.f) {
			continue;
		}

		filename2multiplier[filename] = multiplier;
	}

	params.lora_state = filename2multiplier;
	shouldUpdateLoraState = true;
}

PackedByteArray Diffusion::start() {
	isRunning = true;

	outputData.clear();

	bool vae_decode_only = true;

	if (input_image_buffer != NULL) {
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
		if (params.height != inputImageHeight || params.width != inputImageWidth) {
			printf("resize input image from %dx%d to %dx%d\n", inputImageWidth, inputImageHeight, params.width, params.height);
			/*int resized_height = params.height;
			int resized_width = params.width;

			uint8_t *resized_image_buffer = (uint8_t *)malloc(resized_height * resized_width * 3);
			if (resized_image_buffer == NULL) {
				fprintf(stderr, "error: allocate memory for resize input image\n");
				
				freeInputBuffers();
				isRunning = false;
				return pngData;
			}
			stbir_resize(input_image_buffer, inputImageWidth, inputImageHeight, 0,
					resized_image_buffer, resized_width, resized_height, 0, STBIR_TYPE_UINT8,
					3, STBIR_ALPHA_CHANNEL_NONE, 0,
					STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,
					STBIR_FILTER_BOX, STBIR_FILTER_BOX,
					STBIR_COLORSPACE_SRGB, nullptr);

			// Save resized result
			free(input_image_buffer);
			input_image_buffer = resized_image_buffer;*/

			freeInputBuffers();
			isRunning = false;
			return outputData;
		}
	}

	sd_ctx_params_t sd_ctx_params = {
		params.model_path.c_str(),
		params.clip_l_path.c_str(),
		params.clip_g_path.c_str(),
		params.t5xxl_path.c_str(),
		params.diffusion_model_path.c_str(),
		params.vae_path.c_str(),
		params.taesd_path.c_str(),
		params.control_net_path.c_str(),
		params.lora_model_dir.c_str(),
		params.embedding_dir.c_str(),
		params.stacked_id_embed_dir.c_str(),
		vae_decode_only,
		params.vae_tiling,
		false,
		params.n_threads,
		params.wtype,
		params.rng_type,
		params.schedule,
		params.clip_on_cpu,
		params.control_net_cpu,
		params.vae_on_cpu,
		params.diffusion_flash_attn,
		params.diffusion_conv_direct,
		params.vae_conv_direct,
		params.chroma_use_dit_mask,
		params.chroma_use_t5_mask,
		params.chroma_t5_mask_pad,
	};

	if (sd_ctx == NULL) sd_ctx = new_sd_ctx(&sd_ctx_params);
	
	if (sd_ctx == NULL) {
		printf("new_sd_ctx_t failed\n");

		freeInputBuffers();
		isRunning = false;
		return outputData;
	}

	if (shouldUpdateLoraState) {
		sd_set_lora_state(sd_ctx, &params.lora_state);
		shouldUpdateLoraState = false;
	}

	sd_image_t *control_image = NULL;

	if (control_image_buffer != NULL) {
		// Resize control image ...
		if (controlImageHeight != params.height || controlImageWidth != params.width) {
			printf("resize control image from %dx%d to %dx%d\n", controlImageWidth, controlImageHeight, inputImageWidth, inputImageHeight);
			/*int resized_height = inputImageHeight;
			int resized_width = inputImageWidth;

			uint8_t *resized_image_buffer = (uint8_t *)malloc(resized_height * resized_width * 3);
			if (resized_image_buffer == NULL) {
				fprintf(stderr, "error: allocate memory for resize input image\n");

				freeInputBuffers();
				isRunning = false;
				return pngData;
			}
			stbir_resize(control_image_buffer, inputImageWidth, inputImageHeight, 0,
					resized_image_buffer, resized_width, resized_height, 0, STBIR_TYPE_UINT8,
					3, STBIR_ALPHA_CHANNEL_NONE, 0,
					STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,
					STBIR_FILTER_BOX, STBIR_FILTER_BOX,
					STBIR_COLORSPACE_SRGB, nullptr);

			// Save resized result
			free(control_image_buffer);
			control_image_buffer = resized_image_buffer;*/

			freeInputBuffers();
			isRunning = false;
			return outputData;
		}

		control_image = new sd_image_t{ (uint32_t)params.width,
			(uint32_t)params.height,
			3,
			control_image_buffer };
		if (params.canny_preprocess) { // apply preprocessor
			control_image->data = preprocess_canny(control_image->data,
					control_image->width,
					control_image->height,
					0.08f,
					0.08f,
					0.8f,
					1.0f,
					false);
		}
	}

	sd_image_t *results = NULL;
	sd_image_t input_image = {};
	sd_image_t mask_image = {};
	sd_guidance_params_t guidance_params = {
		params.cfg_scale,
		params.img_cfg_scale,
		params.min_cfg,
		params.guidance,
		{
			params.skip_layers.data(),
			params.skip_layers.size(),
			params.skip_layer_start,
			params.skip_layer_end,
			params.slg_scale,
		}
	};

	if (input_image_buffer != NULL) {
		input_image = {
			(uint32_t)params.width,
			(uint32_t)params.height,
			3,
			input_image_buffer
		};

		/*if (params.mask_path != "") {
			mask_image_buffer = stbi_load(params.mask_path.c_str(), &maskImageWidth, &maskImageHeight, &maskImageChannels, 1);
		}*/
		if (did_set_mask_image) did_set_mask_image = false;
		else {
			std::vector<uint8_t> default_mask_image_vec(params.width * params.height, 255);
			mask_image_buffer = default_mask_image_vec.data();
			maskImageWidth = inputImageWidth;
			maskImageHeight = inputImageHeight;
		}
		
		mask_image = {
			(uint32_t)params.width,
			(uint32_t)params.height,
			1,
			mask_image_buffer
		};
	}

	sd_img_gen_params_t img_gen_params = {
		params.prompt.c_str(),
		params.negative_prompt.c_str(),
		params.clip_skip,
		guidance_params,
		input_image,
		ref_images.data(),
		(int)ref_images.size(),
		mask_image,
		params.width,
		params.height,
		params.sample_method,
		params.sample_steps,
		params.eta,
		params.strength,
		params.seed,
		params.batch_count,
		control_image,
		params.control_strength,
		params.style_ratio,
		params.normalize_input,
		params.input_id_images_path.c_str(),
	};

	results = generate_image(sd_ctx, &img_gen_params);

	if (results == NULL) {
		printf("generate failed\n");

		freeInputBuffers();
		isRunning = false;
		return outputData;
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
	
	if (results[0].data != NULL) {
		uint8_t *resultsData = results[0].data;
		size_t resultsCount = results[0].channel * results[0].width * results[0].height;
		for (size_t i = 0; i < resultsCount; i++) {
			if (resultsData[i] != 127) {
				resultsData = NULL;
				break;
			}
		}
		if (resultsData == NULL) allocFailCount = 0;
		else allocFailCount++;
		
		int len;
		unsigned char *pngArray = stbi_write_png_to_mem((const unsigned char *)results[0].data, 0, results[0].width, results[0].height, results[0].channel, &len, "");
		
		if (pngArray == NULL) {
			printf("generate failed\n");

			freeInputBuffers();
			isRunning = false;
			return outputData;
		}
		
		outputData.resize(len);
		for (int i = 0; i < outputData.size(); ++i) {
			outputData.set(i, pngArray[i]);
		}

		STBIW_FREE(pngArray);

		free(results[0].data);
		results[0].data = NULL;
	}

	free(results);

	freeInputBuffers();
	isRunning = false;

	return outputData;
}

void Diffusion::freeInputBuffers() {
	if (input_image_buffer != NULL) free(input_image_buffer);
	if (control_image_buffer != NULL) free(control_image_buffer);

	input_image_buffer = NULL;
	control_image_buffer = NULL;
}
void Diffusion::freeModel() {
	if (!isRunning && sd_ctx != NULL) {
		free_sd_ctx(sd_ctx);
		sd_ctx = NULL;
	}

	outputData.clear();
}
