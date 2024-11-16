#include "Detector.h"

int main() {
    // Create ONNX detector
    auto onnxFactory = std::make_unique<ONNXDetectorFactory>();
    App onnxApp(std::move(onnxFactory));
    std::string onnx_path = "./data/best_multi_class.onnx";
    std::string video_path = "./data/07-11-2024-11.52.54.avi";
    onnxApp.run(onnx_path, video_path);

    // // Or create TVM detector
    // auto tvmFactory = std::make_unique<TVMDetectorFactory>();
    // App tvmApp(std::move(tvmFactory));
    // tvmApp.run("./data/model.tvm", "./data/video.avi");

    return 0;
}