#include "predict.hpp"


using namespace std;

Predictor::Predictor(const CNNConfig& config,const std::string& model_dir):cfg(config){
    model=new SimpleConvNet(cfg);
    cout << "load bin" << (model_dir.empty() ?"": model_dir) << endl;
    model->load_weights(model_dir);
}

Predictor::~Predictor(){
    delete model;
}

int Predictor::predict(const Tensor& image){
    Tensor output=model->predict(image);
    int pred_idx=0;
    float max_val=output.data[0];
    for(int i=1;i<cfg.output_size;++i){
        if(output.data[i]>max_val){
            max_val=output.data[i];
            pred_idx=i;
        }
    }
    return pred_idx;
}