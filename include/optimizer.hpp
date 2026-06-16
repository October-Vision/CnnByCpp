#pragma once
#include "tensor.hpp"
#include <cassert>

// SGD 优化器 随机梯度下降
class SGD {
public:
    float lr; 

    // 构造函数
    SGD(float learning_rate = 0.01f);

    void update(Tensor& param, const Tensor& grad);
};