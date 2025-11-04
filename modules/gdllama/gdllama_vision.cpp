
#include "gdllama_vision.h"

static bool didInstantiate = false;
static cv::dnn_superres::DnnSuperResImpl cv_dnn_sr_;

Vision::Vision() {
	if (didInstantiate) {
		printf("There can be only one Vision instance\n");
		return;
	}
	else didInstantiate = true;

	cv_dnn_sr_.readModel("FSRCNN_x2.pb");
	cv_dnn_sr_.setModel("fsrcnn", 2);
}
Vision::~Vision() {
    printf("Vision dealloc\n");
}

void Vision::_bind_methods() {
	ClassDB::bind_method(D_METHOD("initialize"), &Vision::initialize);
}

void Vision::initialize() {
	/*cv::Mat img = cv::imread("1.png");
	if (img.empty()) {
		printf("Can't load image\n");
	}

	std::vector<uchar> pngData;
	std::vector<int> compressionParams = { cv::IMWRITE_PNG_COMPRESSION, 1 };
	bool success = cv::imencode(".png", img, pngData, compressionParams);
	const unsigned char *rawData = pngData.data();

	PackedByteArray inputData;
	inputData.resize(pngData.size());
	uint8_t *inputDataPtr = inputData.ptrw();
	for (int i = 0; i < inputData.size(); ++i) {
		inputDataPtr[i] = rawData[i];
	}

	std::vector<uchar> buffer(inputDataPtr, inputDataPtr + inputData.size());
	cv::Mat img2 = cv::imdecode(buffer, cv::IMREAD_UNCHANGED);
	std::vector<uchar> pngData2;
	success = cv::imencode(".png", img2, pngData2, compressionParams);
	const unsigned char *rawData2 = pngData2.data();

	PackedByteArray outputData;
	outputData.resize(pngData2.size());
	uint8_t *outputDataPtr = outputData.ptrw();
	for (int i = 0; i < outputData.size(); ++i) {
		outputDataPtr[i] = rawData2[i];
	}*/
}

cv::dnn_superres::DnnSuperResImpl Vision::cv_dnn_sr() {
	return cv_dnn_sr_;
}
