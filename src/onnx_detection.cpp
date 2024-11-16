#include "Detector.h"



ONNXDetector::ONNXDetector() : env(ORT_LOGGING_LEVEL_WARNING, "ONNXModel") {}

void ONNXDetector::loadModel(const std::string& modelPath) {
    Ort::SessionOptions options;
    session = Ort::Session(env, modelPath.c_str(), options);
}

cv::Mat ONNXDetector::detect(cv::Mat& image, float confThreshold, float iouThreshold) {
    // Perform detection (implement preprocessing, inference, postprocessing)
}

cv::Mat ONNXDetector::preprocess(cv::Mat& image, std::vector<float>& input_tensor) {
    // Preprocessing code
}

cv::Mat ONNXDetector::postprocess(cv::Mat& image, float* data, std::vector<int64_t> shape,
                                  float confThreshold, float iouThreshold) {
    // Postprocessing code
}

// TVMDetector class will be implemented in a similar manner
