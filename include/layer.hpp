#pragma once
#include "tensor.hpp"
#include "Cnn_utils.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

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

//BN层
class BatchNorm{
    private:
    int channels;
    float eps;
    float momentum;

    //更新参数
    Tensor gamma;
    Tensor beta;
    Tensor dgamma;
    Tensor dbeta;

    //全局均值和方差
    Tensor running_mean;
    Tensor running_var;

    //中间变量
    Tensor x_hat;
    Tensor var;
    Tensor mean;
    Tensor x_minus_mean;

    public:
    BatchNorm(int channels, float eps = 1e-5, float momentum = 0.9);


    //前向传播
    Tensor forward(const Tensor& x, bool is_training = true);

    //反向传播
    Tensor backward(const Tensor& dout);
    //提供接口给Model访问参数用于优化器更新
    Tensor& get_W() { return gamma; }  //gamma映射为W,不动SGD 
    Tensor& get_b() { return beta; }   //beta映射为b
    Tensor& get_dW() { return dgamma; }
    Tensor& get_db() { return dbeta; }

    Tensor& get_running_mean() { return running_mean; }
    Tensor& get_running_var() { return running_var; }
};