#include <opencv2/opencv.hpp>
// #include <tvm/runtime/module.h>
// #include <tvm/runtime/registry.h>
// #include <tvm/runtime/packed_func.h>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>
#include "onnxruntime_cxx_api.h"

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
    Ort::Session session{nullptr};
    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
    std::vector<const char*> inputNames;
    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
    std::vector<const char*> outputNames;
};

// class TVMDetector : public IDetector {
// public:
//     TVMDetector();
//     void loadModel(const std::string& modelPath) override;
//     cv::Mat detect(cv::Mat& image, float confThreshold = 0.4f, float iouThreshold = 0.45f) override;

// protected:
//     cv::Mat preprocess(cv::Mat& image, std::vector<float>& input_tensor) override;
//     cv::Mat postprocess(cv::Mat& image, float* data, std::vector<int64_t> shape,
//                        float confThreshold, float iouThreshold) override;

// private:
//     tvm::runtime::Module mod;
//     DLDevice dev;
//     tvm::runtime::NDArray input_array = tvm::runtime::NDArray::Empty({1, 3, 640, 640}, DLDataType{kDLFloat, 32, 1}, dev);
//     tvm::runtime::NDArray output = tvm::runtime::NDArray::Empty({1, 7, 8400}, DLDataType{kDLFloat, 32, 1}, dev);
//     tvm::runtime::PackedFunc set_input = mod_.GetFunction("set_input");
//     tvm::runtime::PackedFunc run = mod_.GetFunction("run");
//     tvm::runtime::PackedFunc get_output = mod_.GetFunction("get_output");
// };

class ONNXDetectorFactory : public DetectorFactory {
public:
    std::unique_ptr<IDetector> createDetector() override {
        return std::make_unique<ONNXDetector>();
    }
};

// class TVMDetectorFactory : public DetectorFactory {
// public:
//     std::unique_ptr<IDetector> createDetector() override {
//         return std::make_unique<TVMDetector>();
//     }
// };

template <typename T>
class SafeQueue {
    public:
        SafeQueue() : q(), m(), c() {}
        void enqueue(T t) {
            std::lock_guard<std::mutex> lock(m);
            q.push(t);
            c.notify_one();
        }
        bool dequeue(T& t) {
            std::unique_lock<std::mutex> lock(m);
            while (q.empty()) {
                if (finished) return false;
                c.wait(lock);
            }
            t = q.front();
            q.pop();
            return true;
        }
        void setFinished() {
            std::lock_guard<std::mutex> lock(m);
            finished = true;
            c.notify_all();
        }
    private:
        std::queue<T> q;
        mutable std::mutex m;
        std::condition_variable c;
        bool finished = false;
};

class Data {
public:
    Data(std::string& videoPath);
    cv::Mat GetData();
    void WriteData(cv::Mat processedFrame);

private:
    cv::VideoCapture cap;
    cv::VideoWriter out;
};

class App {
public:
    App(std::unique_ptr<DetectorFactory> factory);
    void run(std::string& modelPath, std::string& videoPath);

private:
    std::unique_ptr<DetectorFactory> detectorFactory;
};