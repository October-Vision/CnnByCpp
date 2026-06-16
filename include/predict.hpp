#pragma once
#include <string>
#include <vector>
#include <iostream>
#include "Model.hpp"

class Predictor{
    private:
    CNNConfig cfg;
    SimpleConvNet* model;
    public:
        Predictor(const CNNConfig& config,const std::string& model_dir);
        ~Predictor();

        int predict(const Tensor& image);
    
};


