#ifndef SAFEQUEUE_H
#define SAFEQUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class SafeQueue {
public:
    void enqueue(const T& item);
    bool dequeue(T& item);
    void setFinished();
    bool isFinished() const;

private:
    std::queue<T> queue;
    std::mutex mtx;
    std::condition_variable condVar;
    bool finished = false;
};

#include "SafeQueue.cpp" 
#endif // SAFEQUEUE_H
