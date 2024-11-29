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
#include <cuda_runtime.h>

class IDetector {
public:
    int one;
    virtual ~IDetector() = default;
    virtual std::vector<std::vector<float>> Detect(cv::Mat& image, float conf_threshold = 0.4f, float iou_threshold = 0.45f) = 0;
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
    std::vector<std::vector<float>> Detect(cv::Mat& image, float conf_threshold = 0.4f, float iou_threshold = 0.45f) override;

protected:
    cv::Mat Preprocess(cv::Mat& image, std::vector<float>& input_tensor) ;
    std::vector<std::vector<float>> Postprocess(cv::Mat& image, float* data, std::vector<int64_t> shape, float conf_threshold, float iou_threshold);
    
private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session{nullptr};
    std::vector<Ort::AllocatedStringPtr> input_node_name_allocated_strings;
    std::vector<const char*> input_names;
    std::vector<Ort::AllocatedStringPtr> output_node_name_allocated_strings;
    std::vector<const char*> output_names;
    std::vector<Ort::Value> output_tensors;
};

class TVMDetector : public IDetector {
public:
    TVMDetector();
    void LoadModel(const std::string& modelPath,int ) override;
    std::vector<std::vector<float>> Detect(cv::Mat& image, float conf_threshold = 0.4f, float iou_threshold = 0.4f) override;
    int device_type;
protected:
    cv::Mat Preprocess(cv::Mat& image, tvm::runtime::NDArray& input_array);
    std::vector<std::vector<float>> Postprocess(cv::Mat& image, float* data, int num_detections, float conf_threshold, float iou_threshold);
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
        bool latest(T& t) {
            std::unique_lock<std::mutex> lock(queue_mutex);
            while (data_queue.empty()) {
                if (finished) return false;
                cond_var.wait(lock);
            }
            while(data_queue.size()>1) {
                data_queue.pop();
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
        void clear() {
            while(!data_queue.empty()) {
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
    std::vector<cv::Mat> GetData();
    void WriteData(std::vector<cv::Mat> processed_frames);

private:
    std::vector<cv::VideoCapture> caps;
    int sources = 1;
};

class App {
public:
    App(std::unique_ptr<DetectorFactory> factory);
    void Run(std::string& model_path, std::string& cameras_no, int, int);

private:
    std::unique_ptr<DetectorFactory> detector_factory;
    int frame_count = 0;
    std::vector<std::vector<std::vector<float>>> n_boxes;
    bool wait_until = true;
    int sources = 1;
    bool stop = false;
};

#endif //DETECTOR_H
