#include "tensor.hpp"
#include <iostream>
#include <random>
#include <cassert>

// 默认构造
Tensor::Tensor() {}

// 指定形状并用固定值初始化
Tensor::Tensor(const std::vector<int>& s, float init_val) : shape(s) {
    int total_size = 1;
    for (int dim : shape) total_size *= dim;
    data.assign(total_size, init_val);
}

// 获取张量的总元素个数
int Tensor::size() const {
    return data.size();
}

// 高斯分布初始化
Tensor Tensor::randn(const std::vector<int>& s, float mean, float stddev) {
    Tensor t(s);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(mean, stddev);
    
    for (int i = 0; i < t.size(); ++i) {
        t.data[i] = d(gen);
    }
    return t;
}

// 打印形状
void Tensor::print_shape() const {
    std::cout << "Shape: (";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << (i == shape.size() - 1 ? "" : ", ");
    }
    std::cout << ")" << std::endl;
}

// 矩阵点乘 (Dot Product)
Tensor Tensor::dot(const Tensor& B) const {
    assert(shape.size() == 2 && B.shape.size() == 2 && "Dot product currently supports 2D tensors only.");
    assert(shape[1] == B.shape[0] && "Matrix dimensions do not match for dot product.");

    int M = shape[0];
    int N = shape[1];
    int P = B.shape[1];

    Tensor C({M, P}, 0.0f);

    // i-k-j 循环顺序优化缓存命中率
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < N; ++k) {
            float a_val = data[i * N + k];
            for (int j = 0; j < P; ++j) {
                C.data[i * P + j] += a_val * B.data[k * P + j];
            }
        }
    }
    return C;
}

// 矩阵加法 (广播)
Tensor Tensor::operator+(const Tensor& other) const {
    Tensor result(shape);
    if (shape == other.shape) {
        for (int i = 0; i < size(); ++i) {
            result.data[i] = data[i] + other.data[i];
        }
    } else if (shape.size() == 2 && (other.shape.size() == 1 || (other.shape.size() == 2 && other.shape[0] == 1))) {
        int batch_size = shape[0];
        int features = shape[1];
        assert(other.size() == features && "Broadcast dimensions mismatch.");
        
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < features; ++j) {
                result.data[i * features + j] = data[i * features + j] + other.data[j];
            }
        }
    } else {
        assert(false && "Addition for these shapes is not supported yet.");
    }
    return result;
}

//转置
Tensor Tensor::transpose() const {
    assert(shape.size() == 2 && "二维矩阵转置");
    Tensor res({shape[1], shape[0]});
    for(int i = 0; i < shape[0]; ++i) {
        for(int j = 0; j < shape[1]; ++j) {
            res.data[j * shape[0] + i] = data[i * shape[1] + j];
        }
    }
    return res;
}

// 沿batch维度求和，返回1D tensor
Tensor Tensor::sum_axis0() const {
    assert(shape.size() == 2 && "二维矩阵按列求和");
    Tensor res({shape[1]}, 0.0f); 
    for(int i = 0; i < shape[0]; ++i) {
        for(int j = 0; j < shape[1]; ++j) {
            res.data[j] += data[i * shape[1] + j];
        }
    }
    return res;
}


//改变形状
Tensor Tensor::reshape(const std::vector<int>& new_shape) const{
    int new_size=1;
    for(int s : new_shape) new_size *= s;
    assert(new_size == size() && "Reshapefall: 3");
    
    Tensor res = *this; // 拷贝数据
    res.shape = new_shape; // 更新形状
    return res;
}
//卷积输出维度置换NHWC To NCHW
Tensor Tensor::transpose_NHWC_to_NCHW() const {
    assert(shape.size() == 4 && "必须是 4D 张量");
    int N = shape[0], H = shape[1], W = shape[2], C = shape[3];
    Tensor res({N, C, H, W}, 0.0f);
    
    // 重新排列内存中的数据
    for(int n = 0; n < N; ++n) {
        for(int h = 0; h < H; ++h) {
            for(int w = 0; w < W; ++w) {
                for(int c = 0; c < C; ++c) {
                    res.data[n*C*H*W + c*H*W + h*W + w] = data[n*H*W*C + h*W*C + w*C + c];
                }
            }
        }
    }
    return res;
}
//反操作，适用反向传播
Tensor Tensor::transpose_NCHW_to_NHWC() const {
    assert(shape.size() == 4 && "必须是 4D 张量");
    int N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    Tensor res({N, H, W, C}, 0.0f);
    
    for(int n = 0; n < N; ++n) {
        for(int c = 0; c < C; ++c) {
            for(int h = 0; h < H; ++h) {
                for(int w = 0; w < W; ++w) {
                    res.data[n*H*W*C + h*W*C + w*C + c] = data[n*C*H*W + c*H*W + h*W + w];
                }
            }
        }
    }
    return res;
}

void Tensor::save(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    assert(out.is_open() && "fall5保存失败");
    //把底层的连续内存作为二进制块写入硬盘
    out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    out.close();
}

void Tensor::load(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    assert(in.is_open() && "fall6读取失败");
    //从硬盘直接把二进制块读入内存
    in.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    in.close();
}