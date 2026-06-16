#pragma once
#include<fstream>
#include <vector>

class Tensor {
public:
    std::vector<int> shape;
    std::vector<float> data;

    // 构造函数
    Tensor();
    Tensor(const std::vector<int>& s, float init_val = 0.0f);

    // 基础工具
    int size() const;
    static Tensor randn(const std::vector<int>& s, float mean = 0.0f, float stddev = 1.0f);
    void print_shape() const;

    // 核心数学运算
    Tensor dot(const Tensor& B) const;
    Tensor dot_cuda(const Tensor& B) const;//基于GPU点乘加速
    Tensor operator+(const Tensor& other) const;
    Tensor transpose() const;
    Tensor sum_axis0() const;//batch方向求和,计算偏置梯度

    //修改张量形状
    Tensor reshape(const std::vector<int>& new_shape) const;
    //4D张量维度置换(batch,Height,width,channel)To(batch, channel, Height, width)
    Tensor transpose_NHWC_to_NCHW() const;
    //维度置换反操作(Batch, Channel, Height, Width)To(Batch, Height, Width, Channel)
    Tensor transpose_NCHW_to_NHWC() const;


    //保存权重
    void save(const std::string& filename) const;
    void load(const std::string& filename);
};