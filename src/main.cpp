#include "detector.h"

int main() 
{
    int choice;
    std::cout<< "1--> onnx\n2--> tvm"<<std::endl;
    std::string video_path = "data/video.avi";
    std::cin >> choice;
    if(choice == 1)
    {
        std::cout<<"1-->cpu\n2-->gpu\n";
        std::cin>>choice;
        if(choice == 1)
        {
            auto onnxFactory = std::make_unique<ONNXDetectorFactory>();
            App onnxApp(std::move(onnxFactory));
            std::string onnx_path = "models/onnx_x86/best2.onnx";
            onnxApp.Run(onnx_path, video_path,choice);
        }
        else if(choice == 2)
        {
            auto onnxFactory = std::make_unique<ONNXDetectorFactory>();
            App onnxApp(std::move(onnxFactory));
            std::string onnx_path = "models/onnx_x86/best2.onnx";
            onnxApp.Run(onnx_path, video_path,choice);
        }
    }
    else if(choice==2)
    {
        std::cout<<"1-->cpu\n2-->gpu\n";
        std::cin>>choice;
        if(choice == 1)
        {
            auto tvmFactory = std::make_unique<TVMDetectorFactory>();
            App tvmApp(std::move(tvmFactory));
            std::string tvm_path = "./models/tvm_x86_cpu";
            // std::string tvm_path = "./models/yolov8_tvm_autotuned";
            tvmApp.Run(tvm_path,video_path,choice);
        }
        else if(choice == 2)
        {
            auto tvmFactory = std::make_unique<TVMDetectorFactory>();
            App tvmApp(std::move(tvmFactory));
            std::string tvm_path = "models/tvm_x86_gpu";
            // std::string tvm_path = "./models/yolov8_tvm_autotuned";
            tvmApp.Run(tvm_path,video_path,choice);
        }
    }
    
    return 0;
}