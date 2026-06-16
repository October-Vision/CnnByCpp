#pragma once
#include <string>
#include "tensor.hpp"
#include "layer.hpp"
#include "optimizer.hpp"
//模型类封装
struct CNNConfig {
    int input_channels = 1;
    int input_size = 28;
    int filter_num = 30;
    int filter_size = 5;
    int hidden_size = 100;
    int output_size = 10;
    
    // 训练参数
    int iters_num = 1000;
    int batch_size = 100;
    float learning_rate = 0.05f;
};

class SimpleConvNet {
public:
    // 权重与偏置张量
    Tensor W1, b1, W2, b2, W3, b3;

    //使用指针存储网络层类调用会引发冲突
    Convolution* conv1;
    Relu* relu1;
    Pooling* pool1;
    Affine* affine1;
    Relu* relu2;
    Affine* affine2;
    SoftmaxWithLoss* last_layer;

    // 缓存一些核心维度数据，反向传播要用
    int filter_num;
    int pool_output_size;
    int affine1_input_size;

    SimpleConvNet(const CNNConfig& config);
    ~SimpleConvNet();

    // 核心功能接口
    Tensor predict(const Tensor& x);// 正向推理 
    float forward_loss(const Tensor& x, const Tensor& t);// 正向算Loss
    void backward();// 反向传播算梯度
    void update_weights(SGD& optimizer);// 更新权重
    void save_weights(const std::string& prefix =""); //保存
    void load_weights(const std::string& profix ="");
};