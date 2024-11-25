#include "detector.h"
#include <iostream>
#include <chrono>
#include <atomic>

App::App(std::unique_ptr<DetectorFactory> factory) : detector_factory(std::move(factory)) {}

void App::Run(std::string& model_path, std::string& video_path,int choice) {
    auto detector = detector_factory->createDetector();
    detector->LoadModel(model_path,choice);
    Data data(video_path);

    SafeQueue<cv::Mat> frame_queue;
    SafeQueue<cv::Mat> processed_queue;

    int fps = std::stoi(std::getenv("FPS"));
    const int frame_delay = 1000 / fps;
    auto CaptureTask = [&]() {
        cv::Mat frame;
        auto next_frame_time = std::chrono::steady_clock::now();
        int frame_interval = std::stoi(std::getenv("INFERENCE_INTERVAL"));
        while (true) {
            frame = data.GetData();
            if (frame.empty()) break;
            if (frame_count % frame_interval == 0) {
                frame_queue.enqueue(frame.clone());
                frame_count=0;
            }
            while (wait_until) cv::waitKey(100);
            frame_count++;
            processed_queue.enqueue(frame.clone());
            next_frame_time += std::chrono::milliseconds(frame_delay);
            std::this_thread::sleep_until(next_frame_time);
        }
        frame_queue.setFinished();
        processed_queue.setFinished();
    };

    auto ProcessTask = [&]() {
        cv::Mat frame;
        while (frame_queue.dequeue(frame)) {
            boxes = detector->Detect(frame, std::stof(std::getenv("CONF_THRESHOLD")), std::stof(std::getenv("IOU_THRESHOLD")));
            frame_queue.clear();
            if (wait_until) {
                wait_until = false;
            }
        }
    };

    auto WriteTask = [&]() {
        cv::Mat processed_frame;
        cv::Mat image;
        while (processed_queue.dequeue(processed_frame)) {
            cv::resize(processed_frame, processed_frame, cv::Size(640, 640));
            image = processed_frame.clone();
            int no_of_persons=0;
            int no_of_chairs=0;
            int empty_chairs=detector->DetectionLogic(boxes);
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
                cv::rectangle(processed_frame, cv::Point(left, top), cv::Point(right, bottom), color, 1);
                std::string label = "Score: " + std::to_string(score).substr(0, 4) + " Class : " + std::to_string(class_id);
                cv::putText(processed_frame, label, cv::Point(left, top - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 0, 0);
            }
            // std::string text="No of persons "+std::to_string(no_of_persons);
            // std::string text1="No of chairs "+std::to_string(no_of_chairs);
            // std::string text2="Empty chairs "+std::to_string(empty_chairs);
            // cv::rectangle(image, cv::Point(370, 5), cv::Point(700, 110), (0,0,0), -1);
            // cv:: putText(image, text, cv::Point(400, 30), cv::FONT_HERSHEY_SIMPLEX, 0.9,cv::Scalar(255,255,255), 0,0);
            // cv:: putText(image, text1, cv::Point(400, 60), cv::FONT_HERSHEY_SIMPLEX, 0.9,cv::Scalar(255,255,255), 0,0);
            // cv:: putText(image, text2, cv::Point(400, 90), cv::FONT_HERSHEY_SIMPLEX, 0.9,cv::Scalar(255,255,255), 0,0);
            std::vector<std::string> texts = {
                "No of persons " + std::to_string(no_of_persons),
                "No of chairs " + std::to_string(no_of_chairs),
                "Empty chairs " + std::to_string(empty_chairs)
            };

            cv::rectangle(image, cv::Point(370, 5), cv::Point(700, 110), (0,0,0), -1);

            for(int i = 0; i < texts.size(); i++) {
                cv::putText(image, texts[i], cv::Point(400, 30 + i * 30), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255,255,255), 0, 0);
            }
            cv::Mat output_frame;
            cv::hconcat(image,processed_frame,output_frame);
            data.WriteData(output_frame);
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
    std::string output_path = video_path.substr(0, video_path.length()-4) + "_Detection" + video_path.substr(video_path.length()-4, video_path.length());
    
    if(video_path == "live_stream") cap = cv::VideoCapture(0);
    else cap = cv::VideoCapture(video_path);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open or find the video file!\n";
    }
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    int fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));

    out = cv::VideoWriter(output_path, fourcc, fps, cv::Size(frame_width, frame_height), true);
    if (!out.isOpened())
    {
        std::cerr << "Error: Could not open the output video file for writing!\n";
    }
}

void Data::WriteData(cv::Mat processed_frame) {
    std::cout << "Displaying frame" << std::endl;
    out.write(processed_frame);
    cv::imshow("yolo11 inference", processed_frame);
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
