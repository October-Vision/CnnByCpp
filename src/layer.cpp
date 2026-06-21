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
    //return x.dot(W)+b;
    return x.dot_cuda(W)+b;
}

Tensor Affine::backward(const Tensor& dout){
    //计算上层梯度  dx=dout  * w转置
    //Tensor dx=dout.dot(W.transpose());
    Tensor dx=dout.dot_cuda(W.transpose());
    //当前成权重梯度dw=x转置*dout
    //dW=x.transpose().dot(dout);
    dW=x.transpose().dot_cuda(dout);

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
    //Tensor out = col.dot(col_W) + b;
    Tensor out = col.dot_cuda(col_W) + b;
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
    // Tensor dW_col=col.transpose().dot(dout_col);
    Tensor dW_col=col.transpose().dot_cuda(dout_col);
    dW=dW_col.transpose().reshape(W.shape);//FN C FH FW
    //findout tidu
    // Tensor dcol=dout_col.dot(col_W.transpose());
    Tensor dcol=dout_col.dot_cuda(col_W.transpose());
    Tensor dx=col2im(dcol,x.shape,FH,FW,stride,pad);

    return dx;

}   


//池化层实现
Pooling::Pooling(int pool_h,int pool_w,int s,int p)
    : pool_h(pool_h), pool_w(pool_w), stride(s), pad(p) {}

Tensor Pooling::forward(const Tensor& input){
    x=input;
    int N=x.shape[0];
    int C=x.shape[1];
    int H=x.shape[2];
    int W=x.shape[3];

    //计算池化后的输出尺寸
    int out_h=(H+2*pad - pool_h)/stride+1;
    int out_w=(W+2*pad - pool_w)/stride+1;

    //展开
    Tensor col=im2col(x,pool_h,pool_w,stride,pad);

    //这里给了个很好的方法：使用内存排序特性，按通道切分窗口来找最大值；
    int num_windows=N*out_h*out_w*C;//总的窗口数量
    int window_size=pool_h*pool_w;//每个窗口有多少元素

    Tensor out_flat({num_windows},0.0f);
    argmax.resize(num_windows);

    //遍历窗口寻找max和argmax
    for (int i =0; i<num_windows; ++i) {
        float max_val=-1e20f; // 极小值
        int max_idx =-1;
        for (int j = 0; j < window_size; ++j) {
            float val = col.data[i *window_size + j];
            if (val > max_val) {
                max_val = val;
                max_idx = j;
            }
        }
        out_flat.data[i] = max_val; // 保存最大值
        argmax[i] = max_idx;        // 保存最大值在窗口中的相对位置
    }

    //将1D变回4D特征
    //N, out_h, out_w, C
    Tensor out_4d=out_flat.reshape({N,out_h,out_w,C});
    return out_4d.transpose_NHWC_to_NCHW();

}

Tensor Pooling::backward(const Tensor& dout){
    //梯度 N,C,OH,OW置换为N,OH,OW,C匹配正向时候的平铺结构
    Tensor dout_trans=dout.transpose_NCHW_to_NHWC();

    int N=x.shape[0];
    int C=x.shape[1];
    int out_h=dout.shape[2];
    int out_w=dout.shape[3];

    int num_windows=N*out_h*out_w*C;//总的窗口数量
    int window_size=pool_h*pool_w;//每个窗口有多少元素

    //声明全0展开矩阵
    Tensor dcol({N*out_h*out_w, C*window_size}, 0.0f);
    //根据前向记录Argmax,把梯度放回原来位置 no 最大其余职位0
    for (int i = 0; i <num_windows; ++i) {
        int max_idx=argmax[i];
        dcol.data[i*window_size + max_idx] = dout_trans.data[i];
    }

    //折叠回图像
    Tensor dx=col2im(dcol,x.shape,pool_h,pool_w,stride,pad);
    return dx;
}

//BN层实现
BatchNorm::BatchNorm(int channels,float eps,float momentum)
    : channels(channels),eps(eps),momentum(momentum) {
    gamma = Tensor({channels}, 1.0f);
    beta = Tensor({channels}, 0.0f);
    dgamma = Tensor({channels}, 0.0f);
    dbeta = Tensor({channels}, 0.0f);

    running_mean = Tensor({channels}, 0.0f);
    running_var = Tensor({channels}, 1.0f);
}

Tensor BatchNorm::forward(const Tensor& x, bool is_training) {
    int N = x.shape[0];
    int C = x.shape[1];
    int H = x.shape[2];
    int W = x.shape[3];
    assert(C == channels && "BNError:通道数不匹配");

    Tensor out = Tensor(x.shape, 0.0f);
    int m = N * H * W; //元素总数

    if (is_training) {
        mean = Tensor({C}, 0.0f);
        var = Tensor({C}, 0.0f);
        x_minus_mean = Tensor(x.shape, 0.0f);
        x_hat = Tensor(x.shape, 0.0f);

        //每个通道的均值
        for (int n = 0; N > n; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        int idx = ((n * C + c) * H + h) * W + w;
                        mean.data[c] += x.data[idx];
                    }
                }
            }
        }
        for (int c = 0; c < C; ++c) mean.data[c] /= m;

        //方差
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        int idx = ((n * C + c) * H + h) * W + w;
                        float diff = x.data[idx] - mean.data[c];
                        x_minus_mean.data[idx] = diff;
                        var.data[c] += diff * diff;
                    }
                }
            }
        }
        for (int c = 0; c < C; ++c) var.data[c] /= m;
        //更新全局滚动均值和方差
        for (int c = 0; c < C; ++c) {
            running_mean.data[c] = momentum * running_mean.data[c] + (1.0f - momentum) * mean.data[c];
            running_var.data[c] = momentum * running_var.data[c] + (1.0f - momentum) * var.data[c];
        }
    } else {
        mean = running_mean;
        var = running_var;
    }

    for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                float inv_std = 1.0f / std::sqrt(var.data[c] + eps);
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        int idx = ((n * C + c) * H + h) * W + w;
                        float normalized_val;
                        if (is_training) {
                            x_hat.data[idx] = x_minus_mean.data[idx] * inv_std;
                            normalized_val = x_hat.data[idx];
                        } else {
                            normalized_val = (x.data[idx] - mean.data[c]) * inv_std;
                        }
                        // 乘上 gamma，加上 beta
                        out.data[idx] = gamma.data[c] * normalized_val + beta.data[c];
                    }
                }
            }
        }
    return out;
}

Tensor BatchNorm::backward(const Tensor& dout) {
    int N = dout.shape[0];
    int C = dout.shape[1];
    int H = dout.shape[2];
    int W = dout.shape[3];
    int m = N * H * W;

    Tensor dx = Tensor(dout.shape, 0.0f);
    
    // 重置梯度
    fill(dgamma.data.begin(), dgamma.data.end(), 0.0f);
    fill(dbeta.data.begin(), dbeta.data.end(), 0.0f);

    //计算dgamma dbeta
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int idx = ((n * C + c) * H + h) * W + w;
                    dgamma.data[c] += dout.data[idx] * x_hat.data[idx];
                    dbeta.data[c] += dout.data[idx];
                }
            }
        }
    }

    //链式求导
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            float inv_std = 1.0f / std::sqrt(var.data[c] + eps);
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int idx = ((n * C + c) * H + h) * W + w;
                    
                    float dx_hat = dout.data[idx] * gamma.data[c];
                    float dvar = dx_hat * x_minus_mean.data[idx] * -0.5f * inv_std * inv_std * inv_std;
                    
                    //由于跨Batch累加非常复杂,采用数学闭式解推导
                    dx.data[idx] = inv_std * (dx_hat - dgamma.data[c] * x_hat.data[idx] / m - dbeta.data[c] / m);
                }
            }
        }
    }
    return dx;
}