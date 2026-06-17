#pragma once
#include "tensor.hpp"

class Metrics{
    private:
    public:
    // predictions:网络输出的预测概率矩阵(Batch, Classes)
    // labels:One-Hot编码的真实标签矩阵(Batch, Classes)
    static float accuracy(const Tensor& predictions, const Tensor& labels);
    //MSE
    static float mse(const Tensor& predictions, const Tensor& labels);
    //Precision
    static float precision(const Tensor& predictions, const Tensor& labels);
    //Recall
    static float recall(const Tensor& predictions, const Tensor& labels);
    //F1-score
    static float f1_score(const Tensor& predictions, const Tensor& labels);
};