#include "Detector.h"

void TVMDetector::loadModel(const std::string& modelPath) {
    
}

cv::Mat TVMDetector::detect(cv::Mat& image, float confThreshold, float iouThreshold) {
    
}

cv::Mat TVMDetector::preprocess(cv::Mat& image, std::vector<float>& input_tensor) {
    
}

cv::Mat TVMDetector::postprocess(cv::Mat& image, float* data, std::vector<int64_t> shape, 
                                 float confThreshold, float iouThreshold) {
    
    
}
