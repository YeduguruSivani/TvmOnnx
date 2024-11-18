
#include "Detector.h"
using namespace std;
// DLDevice dev;
tvm::runtime::Module mod;
DLDevice dev{static_cast<DLDeviceType>(kDLCPU), 0};
    
TVMDetector::TVMDetector(){}

void TVMDetector::loadModel(const std::string& modelPath) {
   
    int device_type = kDLCPU;
    int device_id = 0;
    mod = tvm::runtime::Module::LoadFromFile(modelPath + "/mod.so");

    ifstream json_in(modelPath + "/mod.json");
    string json_data((istreambuf_iterator<char>(json_in)), istreambuf_iterator<char>());
    json_in.close();

    tvm::runtime::Module mod_executor = (*tvm::runtime::Registry::Get("tvm.graph_executor.create"))(json_data, mod, device_type, device_id);

    ifstream params_in(modelPath + "/mod.params", ios::binary);
    string params_data((istreambuf_iterator<char>(params_in)), istreambuf_iterator<char>());
    params_in.close();

    TVMByteArray params_arr{params_data.c_str(), params_data.size()};
    mod_executor.GetFunction("load_params")(params_arr);

    mod = mod_executor;
}

float TVMDetector:: Iou(const std::vector<float> &boxA, const std::vector<float> &boxB)
{
    const float eps = 1e-6;
    float areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
    float areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);
    float x1 = std::max(boxA[0], boxB[0]);
    float y1 = std::max(boxA[1], boxB[1]);
    float x2 = std::min(boxA[2], boxB[2]);
    float y2 = std::min(boxA[3], boxB[3]);
    float w = std::max(0.f, x2 - x1);
    float h = std::max(0.f, y2 - y1);
    float inter = w * h;
    return inter / (areaA + areaB - inter + eps);
}

void TVMDetector::Nms(std::vector<std::vector<float>> &boxes, const float iou_threshold)
{
    std::sort(boxes.begin(), boxes.end(), [](const std::vector<float> &boxA, const std::vector<float> &boxB)
              { return boxA[4] > boxB[4]; });
    for (int i = 0; i < boxes.size(); ++i)
    {
        if (boxes[i][4] == 0.f)
            continue;
        for (int j = i + 1; j < boxes.size(); ++j)
        {
            if (boxes[i][5] != boxes[j][5])
                continue;
            if (Iou(boxes[i], boxes[j]) > iou_threshold)
                boxes[j][4] = 0.f;
        }
    }
    boxes.erase(std::remove_if(boxes.begin(), boxes.end(), [](const std::vector<float> &box)
                               { return box[4] == 0.f; }),
                boxes.end());
}
cv::Mat TVMDetector::detect(cv::Mat& image, float confThreshold, float iouThreshold) 
{
    cv::Mat resized_frame;

    tvm::runtime::NDArray input_array = tvm::runtime::NDArray::Empty({1, 3, 640, 640}, DLDataType{kDLFloat, 32, 1}, dev);
    tvm::runtime::NDArray output = tvm::runtime::NDArray::Empty({1, 7, 8400}, DLDataType{kDLFloat, 32, 1}, dev);
    tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
    tvm::runtime::PackedFunc run = mod.GetFunction("run");
    tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");

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
    int num_detections = output->shape[2];
    return postprocess(resized_frame, data, num_detections, confThreshold, iouThreshold);
}

cv::Mat TVMDetector::preprocess(cv::Mat& frame, tvm::runtime::NDArray& input_array) 
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
    return resized_frame;
}

cv::Mat TVMDetector::postprocess(cv::Mat& image, float* data, int num_detections,
                                 float confThreshold, float iouThreshold) {
    
    vector<vector<float>> boxes;
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
    Nms(boxes, iou_threshold);
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
        cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 0, 0), 1);
        string label = "Score: " + to_string(score).substr(0, 4) + " Class : " + to_string(class_id);
        cv::putText(image, label, cv::Point(left, top - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 0);
    }
    cv::Mat output_frame;
    image.convertTo(output_frame, CV_8U, 255.0);
    return output_frame;
}
