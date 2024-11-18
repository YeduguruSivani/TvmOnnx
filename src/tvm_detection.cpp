
#include "detector.h"
using namespace std;
tvm::runtime::Module mod;
DLDevice dev;

TVMDetector::TVMDetector(){}

void TVMDetector::LoadModel(const std::string& modelPath,int choice) {
    int device_type ;
    if(choice == 1)
    {
        dev = {static_cast<DLDeviceType>(kDLCPU), 0};
        device_type = kDLCPU;
            
    }
    else if(choice == 2)
    {
        dev = {static_cast<DLDeviceType>(kDLGPU), 0};
        device_type = kDLGPU;
    }
    else 
    {
        throw std::invalid_argument("Invalid choice: must be 1 (CPU) or 2 (GPU)");
    }
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

cv::Mat TVMDetector::Detect(cv::Mat& image, float confThreshold, float iouThreshold) 
{
    cv::Mat resized_frame;

    tvm::runtime::NDArray input_array = tvm::runtime::NDArray::Empty({1, 3, 640, 640}, DLDataType{kDLFloat, 32, 1}, dev);
    tvm::runtime::NDArray output = tvm::runtime::NDArray::Empty({1, 7, 8400}, DLDataType{kDLFloat, 32, 1}, dev);
    tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
    tvm::runtime::PackedFunc run = mod.GetFunction("run");
    tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");

    resized_frame = Preprocess(image, input_array);
    auto start_time = std::chrono::high_resolution_clock::now();
    set_input("images", input_array);
    run();
    get_output(0, output);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time);
    std::cout<<"time taken :"<< duration.count()<<std::endl; 
    int output_size = 1;

    for (int i = 0; i < output->ndim; ++i) 
    {
        output_size *= output->shape[i];
    }

    float* data = new float[output_size];
    float* output_data = static_cast<float*>(output->data);

    output.CopyToBytes(data, output_size * sizeof(float));
    int num_detections = output->shape[2];
    return Postprocess(resized_frame, data, num_detections, confThreshold, iouThreshold);
}

cv::Mat TVMDetector::Preprocess(cv::Mat& frame, tvm::runtime::NDArray& input_array) 
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

cv::Mat TVMDetector::Postprocess(cv::Mat& image, float* data, int num_detections,
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
    BoundariesLogic(boxes);
    std::cout << "Number of boxes :" << boxes.size() << std::endl;
    int no_of_persons=0;
    int num_of_chairs=0;
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
            num_of_chairs++;
            color = cv::Scalar(0, 0, 255);
        }
        if(class_id==1 || class_id==3){
            no_of_persons++;
        }
        cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), color, 1);
        std::string label = "Score: " + std::to_string(score).substr(0, 4) + " Class : " + std::to_string(class_id);
        cv::putText(image, label, cv::Point(left, top - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 0, 0);
    }
    int no_of_people1=0;
    int no_of_chairs1=0;
    people_count.push_back(no_of_persons);
    chair_count.push_back(num_of_chairs);
    if(people_count.size()>=30){
        int k=mode(people_count);
        int l=mode(chair_count);
        people_count.erase(people_count.begin());
        chair_count.erase(chair_count.begin());
        no_of_people1=k;
        no_of_chairs1=l;
    }
    std::string text="no_of_persons "+std::to_string(no_of_people1);
    std::string text1="no_of_chairs "+std::to_string(no_of_chairs1);
    cv:: putText(image, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,cv::Scalar(0, 255, 255), 0,0);
    cv:: putText(image, text1, cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 0.7,cv::Scalar(255, 0, 255), 0,0);
    std::cout<<"number of persons : in the frame "<<no_of_persons<<std::endl;
    std::cout<<"number of chairs : in the frame "<<num_of_chairs<<std::endl;
    std::cout<<"number of persons : in the frame "<<no_of_persons<<std::endl;
    cv::Mat output_frame;
    image.convertTo(output_frame, CV_8U, 255.0);
    boxes.clear();

    return output_frame;
}
