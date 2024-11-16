
#include "Detector.h"


TVMDetector::TVMDetector() : dev{DLDeviceType::kDLCPU, 0} {}

void TVMDetector::loadModel(const std::string& modelPath) {

    dev = {static_cast<DLDeviceType>(device_type), device_id};
    mod = tvm::runtime::Module::LoadFromFile(lib_path + "/mod.so");

    ifstream json_in(lib_path + "/mod.json");
    string json_data((istreambuf_iterator<char>(json_in)), istreambuf_iterator<char>());
    json_in.close();

    tvm::runtime::Module mod_executor = (*tvm::runtime::Registry::Get("tvm.graph_executor.create"))(json_data, mod, device_type, device_id);

    ifstream params_in(lib_path + "/mod.params", ios::binary);
    string params_data((istreambuf_iterator<char>(params_in)), istreambuf_iterator<char>());
    params_in.close();

    TVMByteArray params_arr{params_data.c_str(), params_data.size()};
    mod_executor.GetFunction("load_params")(params_arr);

    mod = mod_executor;
}

cv::Mat TVMDetector::detect(cv::Mat& image, float confThreshold, float iouThreshold) 
{
    cv::Mat resized_frame;
   
    resized_frame = preprocess(image, input_array);
    
    set_input("images", input_array);
    run();
    get_output(0, output);
    int output_size = 1;

    for (int i = 0; i < output->ndim; ++i) 
    {
        output_size *= output->shape[i];
    }

    float* data = new float[output_size];
    float* output_data = static_cast<float*>(output->data);

    output.CopyToBytes(data, output_size * sizeof(float));

    return postprocess(image, data, output, confThreshold, iouThreshold);
}

cv::Mat TVMDetector::preprocess(cv::Mat& image, tvm::runtime::NDArray& input_array) 
{
    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(640, 640));
    resized_frame.convertTo(resized_frame, CV_32F, 1.0 / 255);

    vector<cv::Mat> channels(3);
    vector<float> input_tensor_values;
    cv::split(resized_frame, channels);
    for (auto& ch : channels)
    {
        input_tensor_values.insert(input_tensor_values.end(), (float*)ch.datastart, (float*)ch.dataend);
    }
    input_array.CopyFromBytes(input_tensor_values.data(), input_tensor_values.size() * sizeof(float));
    return resizedImage;
}

cv::Mat TVMDetector::postprocess(cv::Mat& image, float* data, std::vector<int64_t> output,
                                 float confThreshold, float iouThreshold) {
    vector<vector<float>> boxes;
    int num_detections = output->shape[2];
    for (int i = 0; i < num_detections; ++i) 
    {
        float x1 = data[0 * num_detections + i];
        float y1 = data[1 * num_detections + i];
        float x2 = data[2 * num_detections + i];
        float y2 = data[3 * num_detections + i];
        float score_1 = round(data[i + num_detections * 4] * 100) / 100.0;
        float score_2 = round(data[i + num_detections * 5] * 100) / 100.0;
        float score_3 = round(data[i + num_detections * 6] * 100) / 100.0;
        float score_threshold = 0.1;
        if (score_1 > score_threshold || score_2 > score_threshold || score_3 > score_threshold)
        {
            cout << "Detection " << i + 1 << ": "
                        << "x1=" << (x1) << ", y1=" << (y1) << ", x2=" << (x2)
                        << ", y2=" << (y2) << ", score_1 =" << (score_1) << ", score_2 =" << (score_2) << ", score_3 =" << (score_3) << endl;
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

            int left = static_cast<int>(x1 - x2 / 2);
            int top = static_cast<int>(y1 - y2 / 2);
            int right = static_cast<int>(x1 + x2/ 2);
            int bottom = static_cast<int>(y1 + y2 / 2);
            boxes.push_back({static_cast<float>(left), static_cast<float>(top), static_cast<float>(right), static_cast<float>(bottom), max_score, static_cast<float>(class_id)}); // 0 is a placeholder for class_id

        }
    }
    float iou_threshold = 0.3;
    nms(boxes, iou_threshold);
    for (const auto& box : boxes)
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
            color = cv::Scalar(0, 0, 255);
        }
        cv::rectangle(resized_frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 0, 0), 1);
        string label = "Score: " + to_string(score).substr(0, 4) + " Class : " + to_string(class_id);
        cv::putText(resized_frame, label, cv::Point(left, top - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 0);
    }



    return result;
}
