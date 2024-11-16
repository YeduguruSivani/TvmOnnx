#ifndef DETECTOR_H
#define DETECTOR_H

#include <opencv2/opencv.hpp>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <onnxruntime/core/providers/tensorflow/tensorflow_provider.h>
#include <string>
#include <vector>
#include <memory>

class IDetector {
public:
    virtual ~IDetector() = default;
    virtual cv::Mat detect(cv::Mat& image, float confThreshold = 0.4f, float iouThreshold = 0.45f) = 0;
    virtual void loadModel(const std::string& modelPath) = 0;
protected:
    virtual cv::Mat preprocess(cv::Mat& image, std::vector<float>& input_tensor) = 0;
    virtual cv::Mat postprocess(cv::Mat& image, float* data, std::vector<int64_t> shape,
                                float confThreshold, float iouThreshold) = 0;
};

class DetectorFactory {
public:
    virtual ~DetectorFactory() = default;
    virtual std::unique_ptr<IDetector> createDetector() = 0;
};

class ONNXDetector : public IDetector {
public:
    ONNXDetector();
    void loadModel(const std::string& modelPath) override;
    cv::Mat detect(cv::Mat& image, float confThreshold = 0.4f, float iouThreshold = 0.45f) override;

protected:
    cv::Mat preprocess(cv::Mat& image, std::vector<float>& input_tensor) override;
    cv::Mat postprocess(cv::Mat& image, float* data, std::vector<int64_t> shape,
                       float confThreshold, float iouThreshold) override;

private:
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    Ort::Session session;
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
};

class TVMDetector : public IDetector {
public:
    TVMDetector();
    void loadModel(const std::string& modelPath) override;
    cv::Mat detect(cv::Mat& image, float confThreshold = 0.4f, float iouThreshold = 0.45f) override;

protected:
    cv::Mat preprocess(cv::Mat& image, std::vector<float>& input_tensor) override;
    cv::Mat postprocess(cv::Mat& image, float* data, std::vector<int64_t> shape,
                       float confThreshold, float iouThreshold) override;

private:
    tvm::runtime::Module mod;
    DLDevice dev;
    tvm::runtime::NDArray input_array = tvm::runtime::NDArray::Empty({1, 3, 640, 640}, DLDataType{kDLFloat, 32, 1}, dev);
    tvm::runtime::NDArray output = tvm::runtime::NDArray::Empty({1, 7, 8400}, DLDataType{kDLFloat, 32, 1}, dev);
    tvm::runtime::PackedFunc set_input = mod_.GetFunction("set_input");
    tvm::runtime::PackedFunc run = mod_.GetFunction("run");
    tvm::runtime::PackedFunc get_output = mod_.GetFunction("get_output");
};

class ONNXDetectorFactory : public DetectorFactory {
public:
    std::unique_ptr<IDetector> createDetector() override {
        return std::make_unique<ONNXDetector>();
    }
};

class TVMDetectorFactory : public DetectorFactory {
public:
    std::unique_ptr<IDetector> createDetector() override {
        return std::make_unique<TVMDetector>();
    }
};

#endif // DETECTOR_H
