#ifndef DETECTOR_H
#define DETECTOR_H

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
#include <ctime>
#include <chrono>
#include <thread>
#include <condition_variable>
#include "onnxruntime_cxx_api.h"
#include <unordered_map>


class IDetector {
public:
    int one;
    virtual ~IDetector() = default;
    virtual std::vector<std::vector<float>> Detect(cv::Mat& image, float confThreshold = 0.4f, float iouThreshold = 0.45f) = 0;
    virtual void LoadModel(const std::string& modelPath, int) = 0;
    virtual int DetectionLogic(std::vector<std::vector<float>> &boxes);
protected:
    virtual float Iou(const std::vector<float> &boxA, const std::vector<float> &boxB);
    virtual void Nms(std::vector<std::vector<float>> &boxes, const float iou_threshold);
    virtual void BoundariesLogic(std::vector<std::vector<float>> &boxes);
};

class DetectorFactory {
public:
    virtual ~DetectorFactory() = default;
    virtual std::unique_ptr<IDetector> createDetector() = 0;
};

class ONNXDetector : public IDetector {
public:
    ONNXDetector();
    void LoadModel(const std::string& modelPath, int) override;
    std::vector<std::vector<float>> Detect(cv::Mat& image, float confThreshold = 0.4f, float iouThreshold = 0.45f) override;

protected:
    cv::Mat Preprocess(cv::Mat& image, std::vector<float>& input_tensor) ;
    std::vector<std::vector<float>> Postprocess(cv::Mat& image, float* data, std::vector<int64_t> shape, float confThreshold, float iouThreshold);
    
private:
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    Ort::Session session{nullptr};
    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
    std::vector<const char*> inputNames;
    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
    std::vector<const char*> outputNames;
    std::vector<Ort::Value> output_tensors;
};

class TVMDetector : public IDetector {
public:
    TVMDetector();
    void LoadModel(const std::string& modelPath,int ) override;
    std::vector<std::vector<float>> Detect(cv::Mat& image, float confThreshold = 0.4f, float iouThreshold = 0.4f) override;

protected:
    cv::Mat Preprocess(cv::Mat& image, tvm::runtime::NDArray& input_array);
    std::vector<std::vector<float>> Postprocess(cv::Mat& image, float* data, int num_detections, float confThreshold, float iouThreshold);
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
        SafeQueue() : data_queue(), queue_mutex(), cond_var() {}
        void enqueue(T t) {
            std::lock_guard<std::mutex> lock(queue_mutex);
            data_queue.push(t);
            cond_var.notify_one();
        }
        bool dequeue(T& t) {
            std::unique_lock<std::mutex> lock(queue_mutex);
            while (data_queue.empty()) {
                if (finished) return false;
                cond_var.wait(lock);
            }
            t = data_queue.front();
            data_queue.pop();
            return true;
        }
        void setFinished() {
            std::lock_guard<std::mutex> lock(queue_mutex);
            finished = true;
            cond_var.notify_all();
        }
	void clear()
	{
	    while(!data_queue.empty())
	    {
            data_queue.pop();
	    }
	}
    private:
        std::queue<T> data_queue;
        mutable std::mutex queue_mutex;
        std::condition_variable cond_var;
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
    void Run(std::string& modelPath, std::string& videoPath,int );

private:
    std::unique_ptr<DetectorFactory> detectorFactory;
    int frame_interval = std::stoi(std::getenv("INFERENCE_INTERVAL"));
    int frame_count = 0;
    std::vector<std::vector<float>> boxes;
    bool wait_until = true;
};

#endif //DETECTOR_H
