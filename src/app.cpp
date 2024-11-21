#include "detector.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>

App::App(std::unique_ptr<DetectorFactory> factory) : detectorFactory(std::move(factory)) {}

void App::Run(std::string& modelPath, std::string& videoPath,int choice) {
    auto detector = detectorFactory->createDetector();
    detector->LoadModel(modelPath,choice);
    Data data(videoPath);

    SafeQueue<cv::Mat> frameQueue;
    SafeQueue<cv::Mat> processedQueue;

    std::atomic<bool> processingDone(false);
    int fps = std::stoi(std::getenv("FPS"));
    const int frameDelay = 1000 / fps;
    auto captureTask = [&]() {
        cv::Mat frame;
        auto nextFrameTime = std::chrono::steady_clock::now();
        while (true) {
            frame = data.GetData();
            if (frame.empty()) break;
            frameQueue.enqueue(frame.clone());
            
            nextFrameTime += std::chrono::milliseconds(frameDelay);
            std::this_thread::sleep_until(nextFrameTime);
        }
        frameQueue.setFinished();
    };

    auto processTask = [&]() {
        cv::Mat frame;
        static bool flag = true;
        while (frameQueue.dequeue(frame)) {
            cv::Mat result = detector->Detect(frame, std::stof(std::getenv("CONF_THRESHOLD")), std::stof(std::getenv("IOU_THRESHOLD")));
             if (flag) {
                frameQueue.clear();
                flag = false;
            }
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
    
    if(videoPath == "live_stream") cap = cv::VideoCapture(0);
    else cap = cv::VideoCapture(videoPath);
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
