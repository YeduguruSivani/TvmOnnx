#include "detector.h"
#include <iostream>
#include <chrono>
#include <atomic>

App::App(std::unique_ptr<DetectorFactory> factory) : detector_factory(std::move(factory)) {}

void App::Run(std::string& model_path, std::string& video_path,int choice) {
    if(video_path.size() == 1) {
        sources = std::stoi(video_path);
    }
    auto detector = detector_factory->createDetector();
    detector->LoadModel(model_path,choice);
    Data data(video_path);

    std::vector<SafeQueue<cv::Mat>> frame_queues(sources);
    std::vector<SafeQueue<cv::Mat>> processed_queues(sources);

    int fps = std::stoi(std::getenv("FPS"));
    const int frame_delay = 1000 / fps;
    auto CaptureTask = [&]() {

        std::vector<cv::Mat> frames(sources);
        auto next_frame_time = std::chrono::steady_clock::now();
        int frame_interval = std::stoi(std::getenv("INFERENCE_INTERVAL"));
        while (true) {
            frames = data.GetData();
            for(int i=0;i<sources;i++)
            {
                if (frames[i].empty()) break;
                if (frame_count % frame_interval == 0) {
                    frame_queues[i].enqueue(frames[i].clone());
                    frame_count=0;
                }
            }
            for(int i=0;i<sources;i++)
            {
                while (wait_until) cv::waitKey(100);
                processed_queues[i].enqueue(frames[i].clone());
            }
            frame_count++;
            next_frame_time += std::chrono::milliseconds(frame_delay);
            std::this_thread::sleep_until(next_frame_time);
        }
        stop = true;
        for (int i=0;i<sources;i++) {
            frame_queues[i].setFinished();
            processed_queues[i].setFinished();
        }
    };

    auto ProcessTask = [&]() {
        std::vector<cv::Mat> frames(sources);
        n_boxes = std::vector<std::vector<std::vector<float>>>(sources);
        while(!stop){
            for(int i=0; i<sources; i++)
            {
                if (frame_queues[i].dequeue(frames[i])) {
                    n_boxes[i] = detector->Detect(frames[i], std::stof(std::getenv("CONF_THRESHOLD")), std::stof(std::getenv("IOU_THRESHOLD")));
                }
                frame_queues[i].clear();
            }
            if (wait_until) {
                wait_until = false;
            }
        }
    };

    auto WriteTask = [&]() {
        std::vector<cv::Mat> processed_frames(sources);
        std::vector<cv::Mat> images(sources);
        while(!stop) {
            for (int i=0;i<sources;i++) {
                if (processed_queues[i].dequeue(processed_frames[i])) {
                    cv::resize(processed_frames[i], processed_frames[i], cv::Size(640, 640));
                    images[i] = processed_frames[i].clone();
                    int no_of_persons=0;
                    int no_of_chairs=0;
                    int empty_chairs=detector->DetectionLogic(n_boxes[i]);
                    for (const auto &box : n_boxes[i])
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
                        cv::rectangle(processed_frames[i], cv::Point(left, top), cv::Point(right, bottom), color, 1);
                        std::string label = "Score: " + std::to_string(score).substr(0, 4) + " Class : " + std::to_string(class_id);
                        cv::putText(processed_frames[i], label, cv::Point(left, top - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 0, 0);
                    }
                    std::vector<std::string> texts = {
                        "No of persons " + std::to_string(no_of_persons),
                        "No of chairs " + std::to_string(no_of_chairs),
                        "Empty chairs " + std::to_string(empty_chairs)
                    };

                    cv::rectangle(processed_frames[i], cv::Point(370, 5), cv::Point(700, 110), (0,0,0), -1);

                    for(int j = 0; j < texts.size(); j++) {
                        cv::putText(processed_frames[i], texts[j], cv::Point(400, 30 + j * 30), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255,255,255), 0, 0);
                    }
                }
            }
            data.WriteData(processed_frames);
        }

    };

    std::thread CaptureThread(CaptureTask);
    std::thread ProcessingThread(ProcessTask);
    std::thread WritingThread(WriteTask);

    CaptureThread.join();
    ProcessingThread.join();
    WritingThread.join();

    std::cout << "Video processing completed successfully." << std::endl;
}

Data::Data(std::string& video_path) {
    std::string cameras = std::getenv("CAMERAS");
    if(video_path.size() == 1) {
        sources = std::stoi(video_path);
        for (int i=0; i<sources; i++) {
            std::cout << "taking :" << cameras[i*2] << std::endl;
            cv::VideoCapture cap = cv::VideoCapture((cameras[i*2]-'0'));
            caps.push_back(cap);
            if (!cap.isOpened())
            {
                std::cerr << "Error: Could not open or find the video file!\n";
            }
        }
    } 
    else {
        caps.push_back(cv::VideoCapture(video_path));
        if (!caps[0].isOpened())
        {
            std::cerr << "Error: Could not open or find the video file!\n";
        }
    }
}

void Data::WriteData(std::vector<cv::Mat> processed_frames) {
    std::cout << "Displaying frame" << std::endl;
    cv::Mat frame = processed_frames[0];
    for (int i=1; i<sources; i++) {
        cv::hconcat(frame, processed_frames[i], frame);
    }
    cv::imshow("yolo11 inference", frame);
    if (cv::waitKey(10) == 'q') {
        for (int i=1; i<sources; i++) {
            caps[i].release();
        }
        return;
    };
}

std::vector<cv::Mat> Data::GetData(){
    std::vector<cv::Mat> frames(sources);
    for (int i=0; i<sources; i++) {
        caps[i].read(frames[i]);
    }
    return frames;

} 
