#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>

// 去掉了 swap_endian 函数，故意不做大小端转换

std::vector<std::vector<uint8_t>> read_mnist_images(const std::string& full_path) {
    std::ifstream file(full_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "fall:1- " << full_path << std::endl;
        exit(1);
    }

    uint32_t magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    
    // 直接读取，不调用任何转换函数
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_images, sizeof(num_images));
    file.read((char*)&num_rows, sizeof(num_rows));
    file.read((char*)&num_cols, sizeof(num_cols));

    std::cout << "=====================================\n";
    std::cout << "幻数 : " << magic_number << "\n";
    std::cout << "图片数量 : " << num_images << "\n";
    std::cout << "行数: " << num_rows << "\n";
    std::cout << "列数 : " << num_cols << "\n";
    std::cout << "======================================\n";

    // 注意：这里由于 num_rows 和 num_cols 是几个亿，乘积会发生严重的整型溢出
    int image_size = num_rows * num_cols;
    std::cout << "image_size: " << image_size << "\n";

    // 这一行会因为尝试分配超过物理极限的内存或由于 image_size 变为负数而直接引发崩溃
    std::vector<std::vector<uint8_t>> images(num_images, std::vector<uint8_t>(image_size));

    for (uint32_t i = 0; i < num_images; ++i) {
        file.read((char*)images[i].data(), image_size);
    }
    return images;
}

std::vector<uint8_t> read_mnist_labels(const std::string& full_path) {
    std::ifstream file(full_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "fall:1- " << full_path << std::endl;
        exit(1);
    }

    uint32_t magic_number = 0, num_items = 0;
    
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_items, sizeof(num_items));

    std::vector<uint8_t> labels(num_items);
    file.read((char*)labels.data(), num_items);

    return labels;
}

// 验证加载是否正确，大小端转换问题
void print_image_ascii(const std::vector<uint8_t>& image, uint8_t label) {
    std::cout << "\n==================================\n";
    std::cout << "真实标签 (Label): " << (int)label << "\n";
    std::cout << "==================================\n";
    
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            uint8_t pixel = image[i * 28 + j];
            if (pixel > 200) std::cout << "@@";
            else if (pixel > 128) std::cout << "%%";
            else if (pixel > 64)  std::cout << "++";
            else if (pixel > 10)  std::cout << "..";
            else std::cout << "  ";
        }
        std::cout << "\n";
    }
    std::cout << "==================================\n\n";
}

int main() {
    try {
        // 根据你之前的目录结构，假设你在 build 下运行，数据集在 ../data 下
        std::string image_path = "../data/train-images-idx3-ubyte";
        std::string label_path = "../data/train-labels-idx1-ubyte";
        
        auto images = read_mnist_images(image_path);
        auto labels = read_mnist_labels(label_path);
        
        // 这行代码永远不会执行到
        print_image_ascii(images[0], labels[0]);
        
    } catch (const std::exception& e) {
        std::cerr << "\n内存分配异常: " << e.what() << "\n";
    }
    return 0;
}