#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>
#include "tensor.hpp"

struct MNISTData {
    std::vector<std::vector<uint8_t>> images;
    std::vector<uint8_t> labels;
};

// 函数声明
std::vector<std::vector<uint8_t>> read_mnist_images(const std::string& full_path);
std::vector<uint8_t> read_mnist_labels(const std::string& full_path);
void print_image_ascii(const std::vector<uint8_t>& image, uint8_t label);

//数据处理方法，归一化和onehot
Tensor normalize_images(const std::vector<std::vector<uint8_t>>& images);
Tensor one_hot_encode(const std::vector<uint8_t>& labels, int num_classes = 10);