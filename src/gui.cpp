#include "gui.hpp"
#include <iostream>

using namespace std;
using namespace cv;

DrawPad::DrawPad(const string& name, int size, int radius, Predictor* pred)
    : window_name(name), canvas_size(size), brush_radius(radius), is_drawing(false), predictor(pred) 
{
    // 初始化纯黑画板
    canvas = Mat::zeros(canvas_size, canvas_size, CV_8UC1);
}

DrawPad::~DrawPad() {
    destroyWindow(window_name);
}


void DrawPad::mouseCallbackStatic(int event, int x, int y, int flags, void* userdata) {
    DrawPad* pad = reinterpret_cast<DrawPad*>(userdata);
    if (pad) {
        pad->handleMouse(event, x, y, flags);
    }
}

// 真正的鼠标事件处理
void DrawPad::handleMouse(int event, int x, int y, int flags) {
    if (event == EVENT_LBUTTONDOWN) {
        is_drawing = true;
    } else if (event == EVENT_LBUTTONUP) {
        is_drawing = false;
    } else if (event == EVENT_MOUSEMOVE && is_drawing) {
        // 画出白色的轨迹
        circle(canvas, Point(x, y), brush_radius, Scalar(255), -1);
    }
}

// 图像预处理：缩放并转为 float 数组
vector<float> DrawPad::preprocessImage(const Mat& img) {
    Mat resized_img;
    // 抗锯齿缩小到 28x28
    resize(img, resized_img, Size(28, 28), 0, 0, INTER_AREA);

    vector<float> flat_img(784);
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            // MNIST 数据需要归一化到 0.0 ~ 1.0 之间
            flat_img[i * 28 + j] = resized_img.at<uchar>(i, j) / 255.0f;
        }
    }
    return flat_img;
}

void DrawPad::run() {
    namedWindow(window_name, WINDOW_AUTOSIZE);
    // 绑定鼠标事件，并将当前对象实例 (this) 传进去
    setMouseCallback(window_name, mouseCallbackStatic, this);
    cout << "空格识别\n";
    cout << "C清空\n";
    cout << "Q退出\n";

    while (true) {
        imshow(window_name, canvas);
        char key = (char)waitKey(10); // 等待按键

        if (key == 'c' || key == 'C') { 
            canvas.setTo(0);
            cout << "画板清空" << endl;
        } 
        else if (key == ' ') { 
            // 执行推理
            vector<float> flat_input = preprocessImage(canvas);
            Tensor img_tensor({1, 1, 28, 28}, 0.0f);
            for (int i = 0; i < 784; ++i) {
                img_tensor.data[i] = flat_input[i];
            }
            int prediction = predictor->predict(img_tensor);
            cout << "你画的数字是:" << prediction<< endl;
        }
        else if (key == 'q' || key == 'Q') { 
            cout << "exit" << endl;
            break;
        }
    }
}