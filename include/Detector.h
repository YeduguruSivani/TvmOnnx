#include <opencv2/opencv.hpp>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/device_api.h>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <fstream>
#include <condition_variable>
#include "onnxruntime_cxx_api.h"

#define kDLGPU 2

class IDetector {
public:
    virtual ~IDetector() = default;
    virtual cv::Mat detect(cv::Mat& image, float confThreshold = 0.4f, float iouThreshold = 0.45f) = 0;
    virtual void loadModel(const std::string& modelPath) = 0;
protected:
    virtual float Iou(const std::vector<float> &boxA, const std::vector<float> &boxB) = 0;
    virtual void Nms(std::vector<std::vector<float>> &boxes, const float iou_threshold) = 0;
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
    cv::Mat preprocess(cv::Mat& image, std::vector<float>& input_tensor) ;
    cv::Mat postprocess(cv::Mat& image, float* data, std::vector<int64_t> shape,
                       float confThreshold, float iouThreshold);
    void Nms(std::vector<std::vector<float>> &boxes, const float iou_threshold)override;
    virtual float Iou(const std::vector<float> &boxA, const std::vector<float> &boxB)override;

private:
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    Ort::Session session{nullptr};
    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
    std::vector<const char*> inputNames;
    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
    std::vector<const char*> outputNames;
};

class TVMDetector : public IDetector {
public:
    TVMDetector();
    void loadModel(const std::string& modelPath) override;
    cv::Mat detect(cv::Mat& image, float confThreshold = 0.4f, float iouThreshold = 0.45f) override;

protected:
    cv::Mat preprocess(cv::Mat& image, tvm::runtime::NDArray& input_array);
    cv::Mat postprocess(cv::Mat& image, float* data, int num_detections,
                       float confThreshold, float iouThreshold);
    void Nms(std::vector<std::vector<float>> &boxes, const float iou_threshold)override;
    virtual float Iou(const std::vector<float> &boxA, const std::vector<float> &boxB)override;


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