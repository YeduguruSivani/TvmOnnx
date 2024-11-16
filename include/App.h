#ifndef APP_H
#define APP_H

#include "Detector.h"
#include "SafeQueue.h"

class Data {
public:
    Data(const std::string& videoPath);
    cv::Mat GetData();
    void WriteData(cv::Mat processedFrame);
private:
    cv::VideoCapture cap;
    cv::VideoWriter out;
};

class App {
public:
    App(std::unique_ptr<DetectorFactory> factory);
    void run(const std::string& modelPath, const std::string& videoPath);

private:
    std::unique_ptr<DetectorFactory> detectorFactory;
};

#endif // APP_H
