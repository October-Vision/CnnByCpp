#include "log.hpp"
#include <iostream>

CSVLogger::CSVLogger(const std::string& filename, const std::string& header) {
    file.open(filename);
    if (file.is_open()) {
        file << header << "\n";
    } else {
        std::cerr << "error:fall::create " << filename << std::endl;
    }
}

CSVLogger::~CSVLogger() {
    if (file.is_open()) {
        file.close();
    }
}

void CSVLogger::log_row(int step, const std::vector<float>& metrics) {
    if (!file.is_open()) return;
    
    file << step; 
    for (float val : metrics) {
        file << "," << val; // 遍历写入传入的所有浮点数指标
    }
    file << "\n";
}