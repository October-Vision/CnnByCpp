#include "Cnn_utils.hpp"

//多维度转1维度实现
Tensor im2col(const Tensor& input_data, int filter_h, int filter_w, int stride, int pad) {
    assert(input_data.shape.size() == 4 && "im2col 输入fall:4(N, C, H, W)");
    
    int N = input_data.shape[0]; // Batch 数量
    int C = input_data.shape[1]; // 通道数
    int H = input_data.shape[2]; // 图像高
    int W = input_data.shape[3]; // 图像宽

    // 计算输出特征图的尺寸
    int out_h = (H + 2 * pad - filter_h) / stride + 1;
    int out_w = (W + 2 * pad - filter_w) / stride + 1;

    // 创建展开后的大矩阵(N * out_h * out_w, C * filter_h * filter_w)
    Tensor col({N * out_h * out_w, C * filter_h * filter_w}, 0.0f);

    //将图像的每个滑动窗口压扁成矩阵的一行
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int out_y = 0; out_y < out_h; ++out_y) {
                for (int out_x = 0; out_x < out_w; ++out_x) {
                    
                    for (int fy = 0; fy < filter_h; ++fy) {
                        for (int fx = 0; fx < filter_w; ++fx) {
                            
                            //映射回原图的坐标 (Padding)
                            int in_y = out_y * stride + fy - pad;
                            int in_x = out_x * stride + fx - pad;

                            //计算在目标2D矩阵col的行号和列号
                            int row = n * out_h * out_w + out_y * out_w + out_x;
                            int col_idx = c * filter_h * filter_w + fy * filter_w + fx;

                            //坐标在有效图像范围内（没有超出 Padding），则拷贝像素值
                            if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                                // 一维数组平铺索引计算
                                int in_idx = n * C * H * W + c * H * W + in_y * W + in_x;
                                col.data[row * col.shape[1] + col_idx] = input_data.data[in_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    return col;
}
//逆操作
Tensor col2im(const Tensor& col, const std::vector<int>& input_shape, int filter_h, int filter_w, int stride, int pad) {
    int N = input_shape[0], C = input_shape[1], H = input_shape[2], W = input_shape[3];
    int out_h = (H + 2 * pad - filter_h) / stride + 1;
    int out_w = (W + 2 * pad - filter_w) / stride + 1;
    
    Tensor img(input_shape, 0.0f); // 初始化全 0
    
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int out_y = 0; out_y < out_h; ++out_y) {
                for (int out_x = 0; out_x < out_w; ++out_x) {
                    for (int fy = 0; fy < filter_h; ++fy) {
                        for (int fx = 0; fx < filter_w; ++fx) {
                            int in_y = out_y * stride + fy - pad;
                            int in_x = out_x * stride + fx - pad;
                            int row = n * out_h * out_w + out_y * out_w + out_x;
                            int col_idx = c * filter_h * filter_w + fy * filter_w + fx;
                            
                            if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                                int in_idx = n*C * H * W + c * H * W + in_y * W + in_x;
                                //同一个像素的梯度必须累加
                                img.data[in_idx] += col.data[row * col.shape[1] + col_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    return img;
}