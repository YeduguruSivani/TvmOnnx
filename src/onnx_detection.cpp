#include "detector.h"

ONNXDetector::ONNXDetector() : env(ORT_LOGGING_LEVEL_WARNING, "ONNXModel") {}

void ONNXDetector::LoadModel(const std::string& model_path,int choice) {
    
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");

    sessionOptions = Ort::SessionOptions();
    int gpu_device_id;
    Ort::SessionOptions session_options;
    if(choice == 2)
    { 
        gpu_device_id = 0;
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = gpu_device_id;
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
    }
    session = Ort::Session(env, model_path.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    inputNodeNameAllocatedStrings.push_back(std::move(input_name));
    inputNames.push_back(inputNodeNameAllocatedStrings.back().get());

    auto output_name = session.GetOutputNameAllocated(0, allocator);
    outputNodeNameAllocatedStrings.push_back(std::move(output_name));
    outputNames.push_back(outputNodeNameAllocatedStrings.back().get());

    std::cout << "Model loaded successfully" << std::endl;
}

cv::Mat ONNXDetector::Detect(cv::Mat& image, float conf_threshold, float iou_threshold) {
    std::vector<float> image_data;
    cv::Mat preprocessed_image = Preprocess(image, image_data);

    std::array<int64_t, 4> input_shape = {1, 3, 640, 640};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, image_data.data(), image_data.size(), input_shape.data(), input_shape.size());

    const char *input_name = "images";
    const char *output_name = "output0";
    std::vector<const char *> output_names = {output_name};
    if (frame_count % frame_interval == 0) {
        output_tensors = session.Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1, output_names.data(), output_names.size());
	frame_count=0;
    }
    if (output_tensors.empty())
    {
        std::cerr << "Error: No output tensor returned from inference." << std::endl;
    }
    frame_count++;
    float *data = output_tensors.front().GetTensorMutableData<float>();
    std::vector<int64_t> shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();

    cv::Mat detection = Postprocess(preprocessed_image, data, shape, conf_threshold, iou_threshold);

    return detection;
}

cv::Mat ONNXDetector::Preprocess(cv::Mat& image, std::vector<float>& input_tensor) {
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(640, 640));
    resizedImage.convertTo(resizedImage, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> channels(3);
    cv::split(resizedImage, channels);
    for (auto &ch : channels)
    {
        input_tensor.insert(input_tensor.end(), (float *)ch.datastart, (float *)ch.dataend);
    }
    return resizedImage;
}

cv::Mat ONNXDetector::Postprocess(cv::Mat& image, float* data, std::vector<int64_t> shape, float conf_threshold, float iou_threshold) {
    
    cv::Mat resized_image = image.clone();
    std::vector<std::vector<float>> boxes;
    for (int i = 0; i < shape[2]; ++i)
    {
        float cx = data[i + shape[2] * 0];
        float cy = data[i + shape[2] * 1];
        float w = data[i + shape[2] * 2];
        float h = data[i + shape[2] * 3];
        float score_1 = round(data[i + shape[2] * 4] * 100) / 100.0;
        float score_2 = round(data[i + shape[2] * 5] * 100) / 100.0;
        float score_3 = round(data[i + shape[2] * 6] * 100) / 100.0;

        if (score_1 > conf_threshold || score_2 > conf_threshold || score_3 > conf_threshold)
        {
            int class_id = 1;
            float max_score = score_1;
            if (score_2 > score_1)
            {
                class_id = 2;
                max_score = score_2;
            }
            if (score_3 > score_2)
            {
                class_id = 3;
                max_score = score_3;
            }
            int left = static_cast<int>(cx - w / 2);
            int top = static_cast<int>(cy - h / 2);
            int right = static_cast<int>(cx + w / 2);
            int bottom = static_cast<int>(cy + h / 2);
            boxes.push_back({static_cast<float>(left), static_cast<float>(top), static_cast<float>(right), static_cast<float>(bottom), max_score, static_cast<float>(class_id)}); // 0 is a placeholder for class_id
        }
    }
    Nms(boxes, iou_threshold);
    BoundariesLogic(boxes);
    std::cout << "Number of boxes :" << boxes.size() << std::endl;
    int no_of_persons=0;
    int no_of_chairs=0;
    int no_of_empty_chairs=0;
    no_of_empty_chairs = DetectionLogic(boxes);
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
        cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), color, 1);
        std::string label = "Score: " + std::to_string(score).substr(0, 4) + " Class : " + std::to_string(class_id);
        cv::putText(image, label, cv::Point(left, top - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 0, 0);
    }

    std::string text="no_of_persons "+std::to_string(no_of_persons);
    std::string text1="no_of_chairs "+std::to_string(no_of_chairs);
    std::string text2="no_of_empty_chairs "+std::to_string(no_of_empty_chairs);
    cv:: putText(resized_image, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.9,cv::Scalar(0, 255, 255), 0,0);
    cv:: putText(resized_image, text1, cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 0.9,cv::Scalar(255, 0, 255), 0,0);
    cv:: putText(resized_image, text2, cv::Point(50, 30), cv::FONT_HERSHEY_SIMPLEX, 0.9,cv::Scalar(255, 0, 0), 0,0);

    std::cout<<"number of persons : in the frame "<<no_of_persons<<std::endl;
    std::cout<<"number of chairs : in the frame "<<no_of_chairs<<std::endl;
    std::cout<<"number of empty chairs  : in the frame "<<no_of_empty_chairs<<std::endl;
    cv::Mat output_frame;
    cv::hconcat(resized_image,image,output_frame);
    output_frame.convertTo(output_frame,CV_8U,255.0);
    boxes.clear();
    return output_frame;
}

