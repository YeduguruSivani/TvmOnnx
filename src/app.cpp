#include "detector.h"
#include <iostream>
#include <chrono>
#include <atomic>

App::App(std::unique_ptr<DetectorFactory> factory) : detector_factory(std::move(factory)) {}

void App::Run(std::string& model_path, std::string& cameras_no, int processing_unit, int model_choice) {
    if(cameras_no.size() == 1) {
        sources = std::stoi(cameras_no);
    }
    auto detector = detector_factory->createDetector();
    detector->LoadModel(model_path, processing_unit);
    Data data(cameras_no);

    std::vector<SafeQueue<cv::Mat>> frame_queues(sources);
    std::vector<SafeQueue<cv::Mat>> processed_queues(sources);

    int fps = std::stoi(std::getenv("FPS"));
    const int frame_delay = 1000 / fps;

    auto CaptureTask = [&]() {
        std::vector<cv::Mat> frames(sources);
        auto next_frame_time = std::chrono::steady_clock::now();
        int frame_interval = 1;
        if (model_choice == 1) {
            if (processing_unit == 1) {
                frame_interval = std::stoi(std::getenv("ONNX_CPU_INFERENCE_INTERVAL"));
            }
            else {
                frame_interval = std::stoi(std::getenv("ONNX_GPU_INFERENCE_INTERVAL"));
            }
        }
        else {
            if (processing_unit == 1) {
                frame_interval = std::stoi(std::getenv("TVM_CPU_INFERENCE_INTERVAL"));
            }
            else {
                frame_interval = std::stoi(std::getenv("TVM_GPU_INFERENCE_INTERVAL"));
            }
        }
        std::cout << "Frame interval :" << frame_interval << std::endl;
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
                    auto start_time = std::chrono::high_resolution_clock::now();
                    n_boxes[i] = detector->Detect(frames[i], std::stof(std::getenv("CONF_THRESHOLD")), std::stof(std::getenv("IOU_THRESHOLD")));
                    auto end_time = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time);
                    std::cout<<"time taken for pipeline :"<< duration.count()<<std::endl;
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
        while(!stop) {
            for (int i=0;i<sources;i++) {
                if (processed_queues[i].dequeue(processed_frames[i])) {
                    cv::resize(processed_frames[i], processed_frames[i], cv::Size(640, 640));
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

Data::Data(std::string& cameras_no) {
    cv::VideoCapture cap;
    std::string cameras = std::getenv("CAMERAS");
    sources = std::stoi(cameras_no);
    int k=0;
    for (int i=0; i<sources; i++) {
        std::string temp = "";
        while (cameras[k]!=','){
            if (k >= cameras.length()) {
                break;
            };
            temp = temp + cameras[k];
            k++;
        }
        k++;
        std::cout << "Using camera :" << temp << std::endl;
        if(temp.length() ==1){
            cap = cv::VideoCapture(temp[0]-'0');
        }
        else{
            cap = cv::VideoCapture(temp);
        }
        caps.push_back(cap);
        if (!cap.isOpened())
        {
            std::cerr << "Error: Could not open or find the video file!\n";
        }
    }
}

void Data::WriteData(std::vector<cv::Mat> processed_frames) {
    cv::Mat frame = processed_frames[0];
    for (int i=1; i<sources; i++) {
        cv::hconcat(frame, processed_frames[i], frame);
    }
    cv::namedWindow("Inference", cv::WINDOW_NORMAL);
    cv::imshow("Inference", frame);
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
