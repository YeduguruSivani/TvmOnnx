#include "Detector.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>

App::App(std::unique_ptr<DetectorFactory> factory) : detectorFactory(std::move(factory)) {}

void App::run(std::string& modelPath, std::string& videoPath) {
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

Data::Data(std::string& videoPath) {
    std::string outputPath = videoPath.substr(0, videoPath.length()-4) + "_Detection" + videoPath.substr(videoPath.length()-4, videoPath.length());
    cap = cv::VideoCapture(videoPath);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open or find the video file!\n";
    }
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    int fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
    out = cv::VideoWriter(outputPath, fourcc, fps, cv::Size(frameWidth, frameHeight), true);
    if (!out.isOpened())
    {
        std::cerr << "Error: Could not open the output video file for writing!\n";
    }
}

void Data::WriteData(cv::Mat processedFrame) {
    std::cout << "Displaying frame" << std::endl;
    out.write(processedFrame);
    cv::imshow("yolo11 inference", processedFrame);
    if (cv::waitKey(10) == 'q') {
        cap.release();
        return;
    };
}

cv::Mat Data::GetData(){
    cv::Mat frame;
    cap.read(frame);
    return frame;
} 