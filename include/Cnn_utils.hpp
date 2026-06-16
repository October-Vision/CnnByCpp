#pragma once
#include "tensor.hpp"
#include <cassert>

//4D转2D矩阵
Tensor im2col(const Tensor& input_data, int filter_h, int filter_w, int stride = 1, int pad = 0);
//逆操作
Tensor col2im(const Tensor& col, const std::vector<int>& input_shape, int filter_h, int filter_w, int stride = 1, int pad = 0);
