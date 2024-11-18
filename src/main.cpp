#include "Detector.h"

int main() {

    int choice;
    std::cout<< "1--> onnx\n2--> tvm"<<std::endl;
    std::string video_path = "/home/ubuntu6/Downloads/07-11-2024-12.25.54.avi";
    std::cin >> choice;
    if(choice == 1)
    {
        auto onnxFactory = std::make_unique<ONNXDetectorFactory>();
        App onnxApp(std::move(onnxFactory));
        std::string onnx_path = "models/onnx_x86/best2.onnx";
        onnxApp.run(onnx_path, video_path);
    }
    else if(choice==2)
    {
        auto tvmFactory = std::make_unique<TVMDetectorFactory>();
        App tvmApp(std::move(tvmFactory));
        // std::string tvm_path = "./models/tvm_x86_gpu";
        std::string tvm_path = "./models/yolov8_tvm_autotuned";
        // std::string video_path ="./data/video.avi";
        tvmApp.run(tvm_path,video_path);
    }

    return 0;
}