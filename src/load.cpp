#include "load.hpp"

// 匿命名空间，相当于 static 函数，仅在本 cpp 文件内可见
namespace {
    uint32_t swap_endian(uint32_t val) {
        return ((val << 24) & 0xff000000) |
               ((val <<  8) & 0x00ff0000) |
               ((val >>  8) & 0x0000ff00) |
               ((val >> 24) & 0x000000ff);
    }
}

std::vector<std::vector<uint8_t>> read_mnist_images(const std::string& full_path) {
    std::ifstream file(full_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "fall:1- " << full_path << std::endl;
        exit(1);
    }

    uint32_t magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = swap_endian(magic_number);
    
    file.read((char*)&num_images, sizeof(num_images));
    num_images = swap_endian(num_images);
    
    file.read((char*)&num_rows, sizeof(num_rows));
    num_rows = swap_endian(num_rows);
    
    file.read((char*)&num_cols, sizeof(num_cols));
    num_cols = swap_endian(num_cols);

    int image_size = num_rows * num_cols;
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
    magic_number = swap_endian(magic_number);
    
    file.read((char*)&num_items, sizeof(num_items));
    num_items = swap_endian(num_items);

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

//归一化
Tensor normalize_images(const std::vector<std::vector<uint8_t>>& images){
    int num_images = images.size();
    int image_size = images[0].size();//28*28 =784
    Tensor X({num_images,image_size});

    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < image_size; ++j) {
            X.data[i * image_size + j] = images[i][j] / 255.0f;
        }
    }
    return X;
}

//one hot热编码
Tensor one_hot_encode(const std::vector<uint8_t>& labels, int num_classes) {
    int num_labels = labels.size();
    Tensor T({num_labels, num_classes}, 0.0f); // 默认全为 0
    
    for (int i = 0; i < num_labels; ++i) {
        T.data[i * num_classes + labels[i]] = 1.0f; // 对应真实标签位置设为 1
    }
    return T;
}