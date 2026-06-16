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