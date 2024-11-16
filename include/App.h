#include <opencv2/opencv.hpp>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <onnxruntime/core/providers/tensorflow/tensorflow_provider.h>
#include <string>
#include <vector>
#include <queue>
#include <memory>
#include <mutex>
#include <condition_variable>

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
    void run(const std::string& modelPath, const std::string& videoPath);

private:
    std::unique_ptr<DetectorFactory> detectorFactory;
};