
#ifndef GDLLAMA_VISION_H
#define GDLLAMA_VISION_H

#include <opencv2/dnn_superres.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "core/object/ref_counted.h"

class Vision : public RefCounted {
	GDCLASS(Vision, RefCounted);

private:

protected:
	static void _bind_methods();

public:
	static cv::dnn_superres::DnnSuperResImpl cv_dnn_sr();
	void initialize();

	Vision();
	~Vision();
};

#endif
