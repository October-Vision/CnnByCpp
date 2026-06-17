#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "predict.hpp"

class DrawPad {
private:
    std::string window_name;
    int canvas_size;
    int brush_radius;
    cv::Mat canvas;
    bool is_drawing;

    Predictor* predictor; // 持有推理器的指针

    // OpenCV 的鼠标回调函数必须是静态的，因此我们需要这个机制来连接对象实例
    static void mouseCallbackStatic(int event, int x, int y, int flags, void* userdata);
    void handleMouse(int event, int x, int y, int flags);

    // 辅助函数：将画板图像转为神经网络需要的扁平数组
    std::vector<float> preprocessImage(const cv::Mat& img);

public:
    // 构造函数：传入窗口名、画板大小、笔刷粗细，以及一个已经准备好的推理器
    DrawPad(const std::string& name, int size, int radius, Predictor* pred);
    ~DrawPad();

    // 启动交互循环
    void run();
};