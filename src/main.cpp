#include "detector.h"

int main() 
{
    int choice;
    std::cout << "1--> onnx\n2--> tvm" << std::endl;
    char* env_path = std::getenv("VIDEO_PATH");
    std::string video_path = std::string(env_path);
    std::cin >> choice;
    if(choice == 1)
    {
        std::cout<<"1-->cpu\n2-->gpu\n";
        std::cin>>choice;
        if(choice == 1)
        {
            auto onnx_factory = std::make_unique<ONNXDetectorFactory>();
            App onnx_app(std::move(onnx_factory));
            std::string onnx_path = std::getenv("ONNX_MODEL_PATH");
            onnx_app.Run(onnx_path, video_path, choice);
        }
        else if(choice == 2)
        {
            auto onnx_factory = std::make_unique<ONNXDetectorFactory>();
            App onnx_app(std::move(onnx_factory));
            std::string onnx_path = std::getenv("ONNX_MODEL_PATH");
            onnx_app.Run(onnx_path, video_path, choice);
        }
    }
    else if(choice==2)
    {
        std::cout<<"1-->cpu\n2-->gpu\n";
        std::cin>>choice;
        if(choice == 1)
        {
            auto tvm_factory = std::make_unique<TVMDetectorFactory>();
            App tvm_app(std::move(tvm_factory));
            std::string tvm_path = std::getenv("TVM_CPU_MODEL_PATH");
            tvm_app.Run(tvm_path, video_path, choice);
        }
        else if(choice == 2)
        {
            auto tvm_factory = std::make_unique<TVMDetectorFactory>();
            App tvm_app(std::move(tvm_factory));
            std::string tvm_path = std::getenv("TVM_GPU_MODEL_PATH");
            tvm_app.Run(tvm_path, video_path, choice);
        }
    }
    
    return 0;
}
