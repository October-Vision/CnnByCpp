#include "optimizer.hpp"

//构造函数初始化学习率
SGD::SGD(float learning_rate) : lr(learning_rate) {}

//梯度下降更新
void SGD::update(Tensor& param, const Tensor& grad) {
    // 安全锁：确保更新的参数和算出来的梯度形状一样
    assert(param.size() == grad.size() && "Optimizer Error 2: 参数和梯度的尺寸不一致");
    
    // W = W - lr * dW
    //L2正泽华 w=w-lr*(dw*weight)
    float weight_decay_lambda=0.001f;
    for (int i = 0; i < param.size(); ++i) {
        //param.data[i] -= lr * grad.data[i];
        param.data[i] -= lr*(grad.data[i]+weight_decay_lambda*param.data[i]);
    }
}