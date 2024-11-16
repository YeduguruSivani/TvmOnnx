#include "SafeQueue.h"
#include "Detector.h"

template <typename T>
void SafeQueue<T>::enqueue(const T& item) {
    std::lock_guard<std::mutex> lock(mtx);
    queue.push(item);
    condVar.notify_one();
}

template <typename T>
bool SafeQueue<T>::dequeue(T& item) {
    std::lock_guard<std::mutex> lock(mtx);
    if (queue.empty()) return false;
    item = queue.front();
    queue.pop();
    return true;
}

template <typename T>
void SafeQueue<T>::setFinished() {
    finished = true;
    condVar.notify_all();
}

template <typename T>
bool SafeQueue<T>::isFinished() const {
    return finished;
}

template class SafeQueue<cv::Mat>;
