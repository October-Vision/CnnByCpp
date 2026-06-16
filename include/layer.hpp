#pragma once
#include "tensor.hpp"
#include "Cnn_utils.hpp"
#include <vector>
#include <cmath>

//ReLU 激活层
class Relu {
    private:
    public:
        std::vector<bool> mask; // 记录 x <= 0 的位置

        Relu() {}
        Tensor forward(const Tensor& x);
        Tensor backward(const Tensor& dout);
};

//Affine 全连接层 (Y = X * W + B)
class Affine {
    private:
    public:
        Tensor W;  // 权重
        Tensor b;  // 偏置
        Tensor x;  // 缓存输入数据反向传播部分
        
        Tensor dW; // 权重的梯度
        Tensor db; // 偏置的梯度

        // 构造函数，初始化权重和偏置
        Affine(const Tensor& weight, const Tensor& bias);

        Tensor forward(const Tensor& input);
        Tensor backward(const Tensor& dout);
};

class SoftmaxWithLoss{
    private:
    public:
    Tensor y;
    Tensor t;
    float loss;

    SoftmaxWithLoss() : loss(0.0f){}

    float forward(const Tensor& x, const Tensor& t_label);

    Tensor backward(float dout=1.0f);

};
//卷积层
class Convolution{
    private:
    public:
    Tensor W; //filitter weight
    Tensor b; //bias
    int stride;
    int pad;

    //huancun
    Tensor x;
    Tensor col;
    Tensor col_W;

    Tensor dW;
    Tensor db;

    Convolution(const Tensor& weight,const Tensor& bias,int stride=1,int pad=0);
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& dout);
};

//池化层
class Pooling{
    private:
    public:
    int pool_h;
    int pool_w;
    int stride;
    int pad;


    Tensor x;
    std::vector<int> argmax;//记录每次池化窗口最大索引，反向传播要用

    Pooling(int pool_h, int pool_w, int stride = 1, int pad = 0);
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& dout);
    
};