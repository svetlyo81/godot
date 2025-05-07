
#ifndef GDLLAMA_DIFFUSION_H
#define GDLLAMA_DIFFUSION_H

#include "core/object/ref_counted.h"

class Diffusion : public RefCounted {
	GDCLASS(Diffusion, RefCounted);

private:
	bool isRunning = false;

	int inputImageChannels = 0;
	int inputImageWidth = 0;
	int inputImageHeight = 0;
	int controlImageChannels = 0;
	int controlImageWidth = 0;
	int controlImageHeight = 0;
	int maskImageChannels = 0;
	int maskImageWidth = 0;
	int maskImageHeight = 0;
	int allocFailCount = 0;

protected:
	static void _bind_methods();

public:
	bool is_running();
	void set_path(const String &modelPath);
	void set_control_path(const String &controlPath);
	void set_param(const String &paramName_, float paramValue);
	void set_prompt(const String &promptString);
	void set_negative_prompt(const String &promptString);
	void set_image(const PackedByteArray &promptImage);
	void set_control_image(const PackedByteArray &promptImage);
	void set_mask_image(const PackedByteArray &maskImage);
	int get_alloc_fail_count();
	PackedByteArray start();
	void freeInputBuffers();
	void freeModel();

	Diffusion();
	~Diffusion();
};

#endif
