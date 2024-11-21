#include "detector.h"
#include <iostream>
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
        int frame_interval = std::stoi(std::getenv("INFERENCE_INTERVAL"));
        while (true) {
            frame = data.GetData();
            if (frame.empty()) break;
            if (frame_count % frame_interval == 0) {
                frameQueue.enqueue(frame.clone());
                frame_count=0;
            }
            while (wait_until) cv::waitKey(100);
            frame_count++;
            processedQueue.enqueue(frame.clone());
            nextFrameTime += std::chrono::milliseconds(frameDelay);
            std::this_thread::sleep_until(nextFrameTime);
        }
        frameQueue.setFinished();
        processedQueue.setFinished();
    };

    auto processTask = [&]() {
        cv::Mat frame;
        while (frameQueue.dequeue(frame)) {
            boxes = detector->Detect(frame, std::stof(std::getenv("CONF_THRESHOLD")), std::stof(std::getenv("IOU_THRESHOLD")));
            if (wait_until) {
                wait_until = false;
            }
        }
    };

    auto writeTask = [&]() {
        cv::Mat processedFrame;
        cv::Mat image;
        while (processedQueue.dequeue(processedFrame)) {
            cv::resize(processedFrame, processedFrame, cv::Size(640, 640));
            image = processedFrame.clone();
            int no_of_persons=0;
            int no_of_chairs=0;
            int no_of_empty_chairs=detector->DetectionLogic(boxes);
            for (const auto &box : boxes)
            {
                int left = static_cast<int>(box[0]);
                int top = static_cast<int>(box[1]);
                int right = static_cast<int>(box[2]);
                int bottom = static_cast<int>(box[3]);
                float score = box[4];
                int class_id = box[5];
                auto color = cv::Scalar(255, 0, 0);
                if (class_id == 1)
                {
                    color = cv::Scalar(0, 255, 0);
                }
                if (class_id == 2)
                {
                    no_of_chairs++;
                    color = cv::Scalar(0, 0, 255);
                }
                if(class_id==1 || class_id==3){
                    no_of_persons++;
                }
                cv::rectangle(processedFrame, cv::Point(left, top), cv::Point(right, bottom), color, 1);
                std::string label = "Score: " + std::to_string(score).substr(0, 4) + " Class : " + std::to_string(class_id);
                cv::putText(processedFrame, label, cv::Point(left, top - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 0, 0);
            }
            std::string text="no_of_persons "+std::to_string(no_of_persons)+"\nno_of_chairs "+std::to_string(no_of_chairs)+"\nno_of_empty_chairs "+std::to_string(no_of_empty_chairs);
            cv:: putText(image, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.9,cv::Scalar(0, 255, 255), 0,0);
            cv::Mat output_frame;
            cv::hconcat(image,processedFrame,output_frame);
            data.WriteData(output_frame);
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
