#include "Model.hpp"

SimpleConvNet::SimpleConvNet(const CNNConfig& config) {
    filter_num = config.filter_num;
    int filter_num_2=config.filter_num*2;

    //维度计算//28*28
    int conv1_output_size =(config.input_size-config.filter_size)/1+1;
    pool_output_size = conv1_output_size/2;

    //INPUT 12x12
    int conv2_output_size=(pool_output_size-config.filter_size)/1+1;//12-5+1=8*8
    int pool2_output_size=conv2_output_size/2;//4*4

    affine1_input_size = filter_num_2 * pool2_output_size * pool2_output_size;//60*4*4
    //随机初始化权重,不能全为0
    //30*1*5*5conv1
    W1 = Tensor::randn({config.filter_num, config.input_channels, config.filter_size, config.filter_size}, 0.0f, 0.01f);
    b1 = Tensor({config.filter_num}, 0.0f);
    //conv2 60*30*5*5
    W_conv2 = Tensor::randn({filter_num_2, config.filter_num, config.filter_size, config.filter_size}, 0.0f, 0.01f);
    b_conv2 = Tensor({filter_num_2}, 0.0f);


    W2 = Tensor::randn({affine1_input_size, config.hidden_size}, 0.0f, 0.01f);
    b2 = Tensor({config.hidden_size}, 0.0f);
    W3 = Tensor::randn({config.hidden_size, config.output_size}, 0.0f, 0.01f);
    b3 = Tensor({config.output_size}, 0.0f);

    //实例化层
    conv1 = new Convolution(W1, b1, 1, 0);
    relu1 = new Relu();
    pool1 = new Pooling(2, 2, 2, 0);
    conv2 = new Convolution(W_conv2, b_conv2, 1, 0);
    relu_conv2 = new Relu();
    pool2 = new Pooling(2, 2, 2, 0);
    affine1 = new Affine(W2, b2);
    relu2 = new Relu();
    affine2 = new Affine(W3, b3);
    last_layer = new SoftmaxWithLoss();

}

//推理层：
Tensor SimpleConvNet::predict(const Tensor& x) {
    Tensor a1 = conv1->forward(x);
    Tensor z1 = relu1->forward(a1);
    Tensor p1 = pool1->forward(z1);
    //进入第二层
    Tensor a_c2 = conv2->forward(p1);
    Tensor z_c2 = relu_conv2->forward(a_c2);
    Tensor p2 = pool2->forward(z_c2);
    
    //自动获取当前Batch的尺寸，防止测试集抓取的数量和训练集不一致
    int current_batch_size = x.shape[0];
    Tensor p2_flat = p2.reshape({current_batch_size, affine1_input_size});
    
    Tensor a2 = affine1->forward(p2_flat);
    Tensor z2 = relu2->forward(a2);
    return affine2->forward(z2);
}

//loss计算
float SimpleConvNet::forward_loss(const Tensor& x, const Tensor& t) {
    Tensor y=predict(x);
    return last_layer->forward(y, t);
}

//反向求导
void SimpleConvNet::backward() {
    float dout = 1.0f;
    Tensor dx = last_layer->backward(dout);


    dx = affine2->backward(dx);
    dx = relu2->backward(dx);
    Tensor dp2_flat = affine1->backward(dx);
    //batch*60*4*4
    int current_batch_size = dp2_flat.shape[0];
    int filter_num_2 = W_conv2.shape[0];
    int pool2_out_size = W_conv2.shape[2] == 5 ? 4 : 4; // 推导已知为 4

    Tensor dp2 = dp2_flat.reshape({current_batch_size, filter_num_2, pool2_out_size, pool2_out_size});
    
    Tensor dz_c2 = pool2->backward(dp2);
    Tensor da_c2 = relu_conv2->backward(dz_c2);
    Tensor dp1 = conv2->backward(da_c2); // 输出给第一层的梯度

    //第一层
    Tensor dz1 = pool1->backward(dp1);
    Tensor da1 = relu1->backward(dz1);
    conv1->backward(da1);


    // // 动态重塑回 4D 传给池化层
    // int current_batch_size = dp1_flat.shape[0];
    // Tensor dp1 = dp1_flat.reshape({current_batch_size, filter_num, pool_output_size, pool_output_size});
    
    // Tensor dz1 = pool1->backward(dp1);
    // Tensor da1 = relu1->backward(dz1);
    // conv1->backward(da1);
}

//权重跟新：
void SimpleConvNet::update_weights(SGD& optimizer) {
    optimizer.update(conv1->W, conv1->dW);
    optimizer.update(conv1->b, conv1->db);
    optimizer.update(conv2->W, conv2->dW);
    optimizer.update(conv2->b, conv2->db);
    optimizer.update(affine1->W, affine1->dW);
    optimizer.update(affine1->b, affine1->db);
    optimizer.update(affine2->W, affine2->dW);
    optimizer.update(affine2->b, affine2->db);
}

void SimpleConvNet::save_weights(const std::string& prefix) {
    conv1->W.save(prefix+"W1.bin");
    conv1->b.save(prefix+"b1.bin");
    conv2->W.save(prefix + "W_conv2.bin"); 
    conv2->b.save(prefix + "b_conv2.bin");
    affine1->W.save(prefix+"W2.bin");
    affine1->b.save(prefix+"b2.bin");
    affine2->W.save(prefix+"W3.bin");
    affine2->b.save(prefix+"b3.bin");
}

//权重加载实现
void SimpleConvNet::load_weights(const std::string& prefix) {
    conv1->W.load(prefix + "W1.bin");
    conv1->b.load(prefix + "b1.bin");
    conv2->W.load(prefix + "W_conv2.bin"); 
    conv2->b.load(prefix + "b_conv2.bin");
    affine1->W.load(prefix + "W2.bin");
    affine1->b.load(prefix + "b2.bin");
    affine2->W.load(prefix + "W3.bin");
    affine2->b.load(prefix + "b3.bin");
}

SimpleConvNet::~SimpleConvNet() {
    delete conv1; delete relu1; delete pool1;
    delete affine1; delete relu2; delete affine2;
    delete conv2; delete relu_conv2; delete pool2;
    delete last_layer;
}