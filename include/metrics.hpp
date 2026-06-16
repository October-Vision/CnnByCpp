#pragma once
#include "tensor.hpp"

class Metrics{
    private:
    public:
    // predictions:网络输出的预测概率矩阵(Batch, Classes)
    // labels:One-Hot编码的真实标签矩阵(Batch, Classes)
    static float accuracy(const Tensor& predictions, const Tensor& labels);
};