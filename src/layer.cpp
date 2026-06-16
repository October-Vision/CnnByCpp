#include "layer.hpp"

//Relu
Tensor Relu::forward(const Tensor& x){
    //每次前向传播重置mask
    mask.assign(x.size(),false);
    Tensor out= x;

    for(int i =0 ;i<x.size();++i){
        if(x.data[i]<=0){
            mask[i]=true;
            out.data[i]=0.0f;  
        }
    }

    return out;
}

Tensor Relu::backward(const Tensor& dout){
    //Relu反向传播部分
    Tensor dx=dout;
    for(int i =0;i<dout.size();++i){
        if(mask[i]){
            dx.data[i]=0.0f;

        }
    }
    return dx;
}

//全连接层实现

Affine::Affine(const Tensor& weight, const Tensor& bias){
    W = weight;
    b = bias;
}
Tensor Affine::forward(const Tensor& input){
    x=input;  //缓存，在反向传播时使用
    return x.dot(W)+b;
}

Tensor Affine::backward(const Tensor& dout){
    //计算上层梯度  dx=dout  * w转置
    Tensor dx=dout.dot(W.transpose());
    //当前成权重梯度dw=x转置*dout
    dW=x.transpose().dot(dout);

    //BIAS 梯度 db=sum(dout,axis=0)
    db=dout.sum_axis0();
    return dx;
}


//Softmaxwithloss实现：
float SoftmaxWithLoss::forward(const Tensor& x, const Tensor& t_label){
    t=t_label;
    y=Tensor(x.shape);

    int batch_size = x.shape[0];
    int num_classes=x.shape[1];
    loss=0.0f;
    for(int i=0;i<batch_size;++i){
        //find max sample C 放置计算移除
        float max_val= x.data[i*num_classes];
        for(int j=0;j<num_classes;++j){
            if(x.data[i*num_classes+j]>max_val){
                max_val = x.data[i*num_classes+j];
            }
        }

        //计算exp（x-max）求和
        float sum_exp=0.0f;
        for (int j = 0; j < num_classes; ++j) {
                    y.data[i*num_classes + j] = std::exp(x.data[i * num_classes + j] - max_val);
                    sum_exp += y.data[i * num_classes + j];
                }
        
        //计算softmax概率和交叉熵误差
        for (int j = 0; j< num_classes; ++j) {
            y.data[i *num_classes+j] /= sum_exp; // 归一化为概率
            // 交叉熵误差仅在正确标签为 1 的地方有值
            // 加1e-7f是为了防止 log(0) 导致 -inf 崩溃
            if (t.data[i * num_classes + j] == 1.0f) {
                loss -= std::log(y.data[i * num_classes + j] + 1e-7f);
            }
        }

    }

    loss /=batch_size;//平均损失
    return loss;


}

//反向传播
Tensor SoftmaxWithLoss::backward(float dout){
    int batch_size=t.shape[0];
    int num_classes  =t.shape[1];
    Tensor dx(t.shape);
    //(y-t)/batch_size
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            dx.data[i * num_classes + j] = (y.data[i * num_classes + j] - t.data[i * num_classes + j]) / batch_size * dout;
        }
    }
    return dx;
}


//Convolution实现
Convolution::Convolution(const Tensor& weight, const Tensor& bias, int s, int p) 
    : W(weight), b(bias), stride(s), pad(p) {}

Tensor Convolution::forward(const Tensor& input){
    x=input;
    int FN=W.shape[0]; //数量
    int C=W.shape[1];   //cha
    int FH=W.shape[2];  //高
    int FW=W.shape[3];  //宽

    int N=x.shape[0];
    int out_h=(x.shape[2]+2*pad-FH)/stride+1;
    int out_w=(x.shape[3]+2*pad-FW)/stride+1;


    //图像展开
    col=im2col(x,FH,FW,stride,pad);
    //滤波器权重展开
    Tensor w_reshaped = W.reshape({FN, C*FH*FW});
    col_W = w_reshaped.transpose();
    //卷积To* +b
    Tensor out = col.dot(col_W) + b;
    //out(N*out_h*out_w,FN)Bianhui N, FN, out_h, out_w
    Tensor out_4d = out.reshape({N, out_h, out_w, FN});
    return out_4d.transpose_NHWC_to_NCHW();

}

Tensor Convolution::backward(const Tensor& dout){
    int FN = W.shape[0];
    int C= W.shape[1];
    int FH = W.shape[2];
    int FW = W.shape[3];

    //维度置换，重塑梯度
    Tensor dout_trans = dout.transpose_NCHW_to_NHWC();
    int row_size = dout_trans.size() / FN; //原张量总大小除以FN计算行数
    Tensor dout_col = dout_trans.reshape({row_size, FN});
    dout_col=dout_trans.reshape({row_size,FN});
    //计算d偏置and dw
    db=dout_col.sum_axis0();
    Tensor dW_col=col.transpose().dot(dout_col);
    dW=dW_col.transpose().reshape(W.shape);//FN C FH FW
    //findout tidu
    Tensor dcol=dout_col.dot(col_W.transpose());
    Tensor dx=col2im(dcol,x.shape,FH,FW,stride,pad);

    return dx;

}   