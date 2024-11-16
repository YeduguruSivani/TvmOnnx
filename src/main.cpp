#include "Detector.h"

int main() {
    // Create ONNX detector
    auto onnxFactory = std::make_unique<ONNXDetectorFactory>();
    App onnxApp(std::move(onnxFactory));
    onnxApp.run("./data/best_multi_class.onnx", "./data/video.avi");

    // Or create TVM detector
    auto tvmFactory = std::make_unique<TVMDetectorFactory>();
    App tvmApp(std::move(tvmFactory));
    tvmApp.run("./data/model.tvm", "./data/video.avi");

    return 0;
}