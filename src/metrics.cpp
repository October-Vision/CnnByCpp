#include "metrics.hpp"

float Metrics::accuracy(const Tensor& predictions, const Tensor& labels) {
    int batch_size = predictions.shape[0];
    int num_classes = predictions.shape[1];
    int correct_count = 0;

    for (int i = 0; i < batch_size; ++i) {
        //预测概率最大的索引
        int pred_idx = 0;
        float max_pred = predictions.data[i * num_classes];
        for (int j = 1; j < num_classes; ++j) {
            if (predictions.data[i * num_classes + j] > max_pred) {
                max_pred = predictions.data[i * num_classes + j];
                pred_idx = j;
            }
        }
        
        //真实标签的索引
        int label_idx = 0;
        for (int j = 0; j <num_classes; ++j) {
            if (labels.data[i * num_classes + j] == 1.0f) { // One-Hot 中 1.0 的位置
                label_idx = j;
                break;
            }
        }
        if (pred_idx == label_idx) {
            correct_count++;
        }
    }
    return (float)correct_count/batch_size;
}
//
float Metrics::mse(const Tensor& predictions,const Tensor& labels){
    int total_elements= predictions.shape[0] *predictions.shape[1];
    float sum_squared_error = 0.0f;

    for (int i = 0; i < total_elements; ++i) {
        float diff = predictions.data[i] - labels.data[i];
        sum_squared_error += diff*diff;
    }
    return sum_squared_error/total_elements;
}
//TP FN FP
static void get_confusion_stats(const Tensor& preds, const Tensor& labels, 
            std::vector<int>& TP, std::vector<int>& FP, std::vector<int>& FN) {
    int batch_size= preds.shape[0];
    int num_classes= preds.shape[1];
    
    TP.assign(num_classes, 0);
    FP.assign(num_classes, 0);
    FN.assign(num_classes, 0);

    for (int i = 0; i < batch_size; ++i) {
        int pred_idx = 0;
        float max_pred = preds.data[i * num_classes];
        for (int j= 1; j< num_classes; ++j) {
            if (preds.data[i*num_classes +j] > max_pred) {
                max_pred = preds.data[i *num_classes + j];
                pred_idx = j;
            }
        }
        
        int label_idx = 0;
        for (int j = 0; j < num_classes; ++j) {
            if (labels.data[i * num_classes + j] == 1.0f) {
                label_idx = j;
                break;
            }
        }

        if (pred_idx == label_idx) {
            TP[label_idx]++;
        } else {
            FP[pred_idx]++;
            FN[label_idx]++;
        }
    }
}

//Precision
float Metrics::precision(const Tensor& predictions, const Tensor& labels) 
{
    int num_classes=predictions.shape[1];
    std::vector<int> TP, FP, FN;
    get_confusion_stats(predictions,labels,TP,FP,FN);

    float macro_precision=0.0f;
    for(int i=0;i<num_classes;++i){
        if(TP[i]+FP[i]>0){
            macro_precision+=(float)TP[i]/(TP[i]+FP[i]);


        }

    }
    return macro_precision/num_classes;
}

float Metrics::recall(const Tensor& predictions,const Tensor& labels){
    int num_classes=predictions.shape[1];
    std::vector<int> TP, FP, FN;
    
    get_confusion_stats(predictions,labels,TP,FP,FN);

    float macro_recall=0.0f;
    for(int i=0;i<num_classes;++i){
        if(TP[i]+FN[i]>0){
            macro_recall+=(float)TP[i]/(TP[i]+FN[i]);


        }

    }
    return macro_recall/num_classes;
}


//F1-score
float Metrics::f1_score(const Tensor& predictions,const Tensor& labels){
    float p=precision(predictions,labels);
    float r=recall(predictions,labels);
    if(p+r==0.0f) return 0;
    return 2*(p*r)/(p+r);
}