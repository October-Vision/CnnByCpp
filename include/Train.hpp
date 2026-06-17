#pragma once
#include <string>
#include <iostream>
#include <random>
#include <iomanip>
#include <fstream>

#include "load.hpp"
#include "metrics.hpp"
#include "Model.hpp"
#include "log.hpp"

class Trainer {
private:
    CNNConfig cfg; //记录训练参数(iters,batch_size)
public:
    Trainer(const CNNConfig& config);
    // 将Model作为引用传进来进行训练
    void fit(SimpleConvNet& model,
        const std::string& train_img_path,
        const std::string& train_lbl_path,
        const std::string& test_img_path,
        const std::string& test_lbl_path);
};