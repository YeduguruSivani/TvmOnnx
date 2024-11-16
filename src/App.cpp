#include "App.h"
#include "SafeQueue.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>

App::App(std::unique_ptr<DetectorFactory> factory) : detectorFactory(std::move(factory)) {}

void App::run(const std::string& modelPath, const std::string& videoPath) {
    auto detector = detectorFactory->createDetector();
    detector->loadModel(modelPath);
    Data data(videoPath);

    SafeQueue<cv::Mat> frameQueue;
    SafeQueue<cv::Mat> processedQueue;

    std::atomic<bool> processingDone(false);

    auto captureTask = [&]() {
        cv::Mat frame;
        while (true) {
            frame = data.GetData();
            if (frame.empty()) break;
            frameQueue.enqueue(frame.clone());
        }
        frameQueue.setFinished();
    };

    auto processTask = [&]() {
        cv::Mat frame;
        while (frameQueue.dequeue(frame)) {
            cv::Mat result = detector->detect(frame);
            processedQueue.enqueue(result);
        }
        processedQueue.setFinished();
    };

    auto writeTask = [&]() {
        cv::Mat processedFrame;
        while (processedQueue.dequeue(processedFrame)) {
            data.WriteData(processedFrame);
        }
    };

    std::thread captureThread(captureTask);
    std::thread processingThread(processTask);
    std::thread writingThread(writeTask);

    captureThread.join();
    processingThread.join();
    writingThread.join();

    std::cout << "Video processing completed successfully." << std::endl;
}

int main() {
    std::unique_ptr<DetectorFactory> factory = std::make_unique<ONNXDetectorFactory>();
    App app(std::move(factory));
    app.run("model.onnx", "video.mp4");

    return 0;
}
