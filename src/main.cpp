#include "detector.h"

int main() 
{
    int model_choice;
    std::cout << "1--> onnx\n2--> tvm" << std::endl;
    std::cin >> model_choice;
    char* env_path = std::getenv("CAMERAS_NO");
    std::string cameras_no = std::string(env_path);
    if(model_choice == 1)
    {
        std::cout<<"1-->cpu\n2-->gpu\n";
        int processing_unit;
        std::cin>>processing_unit;
        if(processing_unit == 1)
        {
            auto onnx_factory = std::make_unique<ONNXDetectorFactory>();
            App onnx_app(std::move(onnx_factory));
            std::string onnx_path = std::getenv("ONNX_MODEL_PATH");
            onnx_app.Run(onnx_path, cameras_no, processing_unit, model_choice);
        }
        else if(processing_unit == 2)
        {
            auto onnx_factory = std::make_unique<ONNXDetectorFactory>();
            App onnx_app(std::move(onnx_factory));
            std::string onnx_path = std::getenv("ONNX_MODEL_PATH");
            onnx_app.Run(onnx_path, cameras_no, processing_unit, model_choice);
        }
    }
    else if(model_choice==2)
    {
        std::cout<<"1-->cpu\n2-->gpu\n";
        int processing_unit;
        std::cin>>processing_unit;
        if(processing_unit == 1)
        {
            auto tvm_factory = std::make_unique<TVMDetectorFactory>();
            App tvm_app(std::move(tvm_factory));
            std::string tvm_path = std::getenv("TVM_CPU_MODEL_PATH");
            tvm_app.Run(tvm_path, cameras_no, processing_unit, model_choice);
        }
        else if(processing_unit == 2)
        {
            auto tvm_factory = std::make_unique<TVMDetectorFactory>();
            App tvm_app(std::move(tvm_factory));
            std::string tvm_path = std::getenv("TVM_GPU_MODEL_PATH");
            tvm_app.Run(tvm_path, cameras_no, processing_unit, model_choice);
        }
    }
    
    return 0;
}
