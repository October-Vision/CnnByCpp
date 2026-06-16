#include "main.hpp"
using namespace std;

// int input_size = 784;      // 输入层：28x28 像素
// int hidden_size = 50;      // 隐藏层：50 个神经元
// int output_size = 10;      // 输出层：10 个分类 (0~9)
    
// int iters_num = 10000;     // 总共训练 10000 次 (Iteration)
// int batch_size = 100;      // 每次抽出 100 张图片 (Mini-batch)
// float learning_rate = 0.1f; // 学习率

/* CNN参数*/
    int input_channels = 1;    // 灰度图单通道
    int input_size = 28;       // 28x28
    int filter_num = 30;       // 卷积核数量
    int filter_size = 5;       // 卷积核尺寸 5x5
    int hidden_size = 100;     // 隐藏层神经元
    int output_size = 10;      // 0-9 分类
    
    int iters_num = 1000;      // 测试跑 1000 次
    int batch_size = 100;      // 每次抓取 100 张图片
    float learning_rate = 0.05f; // 学习率

int baseline(){
    //load
    string train_image_path = "/home/wang/C_cnn/data/train-images-idx3-ubyte";
    string train_label_path = "/home/wang/C_cnn/data/train-labels-idx1-ubyte";
    string test_image_path  = "/home/wang/C_cnn/data/t10k-images-idx3-ubyte";
    string test_label_path  = "/home/wang/C_cnn/data/t10k-labels-idx1-ubyte";

    MNISTData train_dataset;
    train_dataset.images = read_mnist_images(train_image_path);
    train_dataset.labels = read_mnist_labels(train_label_path);
    MNISTData test_dataset;
    test_dataset.images = read_mnist_images(test_image_path);
    test_dataset.labels = read_mnist_labels(test_label_path);
    cout << "实际读到的 训练集 图片数量: " << train_dataset.images.size() << " (期望 60000)" << endl;
    cout << "实际读到的 测试集 图片数量: " << test_dataset.images.size() << " (期望 10000)" << endl;
    //数据预处理
    Tensor x_train = normalize_images(train_dataset.images).reshape({60000, 1, 28, 28});
    Tensor t_train = one_hot_encode(train_dataset.labels);

    Tensor x_test = normalize_images(test_dataset.images).reshape({10000, 1, 28, 28});
    Tensor t_test = one_hot_encode(test_dataset.labels);
    int train_size = x_train.shape[0];

    //计算特征图尺寸
    int conv_output_size = (input_size-filter_size)/1+1; // (28-5)+1 = 24
    int pool_output_size = conv_output_size/2;               // 2x2池化后：12
    int affine1_input_size = filter_num*pool_output_size*pool_output_size; // 30*12*12 = 4320

    //初始化 CNN 网络参数
    Tensor W1 = Tensor::randn({filter_num, input_channels, filter_size, filter_size}, 0.0f, 0.01f);
    Tensor b1({filter_num}, 0.0f);
    
    Tensor W2 = Tensor::randn({affine1_input_size, hidden_size}, 0.0f, 0.01f);
    Tensor b2({hidden_size}, 0.0f);

    Tensor W3 = Tensor::randn({hidden_size, output_size}, 0.0f, 0.01f);
    Tensor b3({output_size}, 0.0f);

    //调用
    Convolution conv1(W1, b1, 1, 0); 
    Relu relu1;
    Pooling pool1(2,2,2,0);//2x2池化
    Affine affine1(W2, b2);
    Relu relu2;
    Affine affine2(W3, b3);
    SoftmaxWithLoss last_layer;
    
    SGD optimizer(learning_rate);

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, train_size - 1);

    //训练：
    for (int i = 0; i < iters_num; ++i) {
        
        // --- 获取 Mini-batch ---
        Tensor x_batch({batch_size, 1, 28, 28}); // 4D
        Tensor t_batch({batch_size, output_size}); // 2D
        
        for (int k = 0; k < batch_size; ++k) {
            int random_idx = dis(gen); 
            copy(x_train.data.begin() + random_idx * 784, 
                 x_train.data.begin() + (random_idx + 1) * 784, 
                 x_batch.data.begin() + k * 784);
            copy(t_train.data.begin() + random_idx * output_size, 
                 t_train.data.begin() + (random_idx + 1) * output_size, 
                 t_batch.data.begin() + k * output_size);
                }
    //forword
        Tensor a1 = conv1.forward(x_batch);   // 输出 4D: (100, 30, 24, 24)
        Tensor z1 = relu1.forward(a1);    // 输出 4D
        Tensor p1=pool1.forward(z1);    //将为
        //4D_TO2D
        Tensor p1_flat = p1.reshape({batch_size, affine1_input_size}); 
        
        Tensor a2 = affine1.forward(p1_flat);    // 输出 2D: (100, 100)
        Tensor z2 = relu2.forward(a2);       // 输出 2D
        Tensor a3 = affine2.forward(z2);        // 输出 2D: (100, 10)
        
        float loss = last_layer.forward(a3, t_batch);

    //backword

        float dout = 1.0f;
        Tensor dx = last_layer.backward(dout);    // (100, 10)
        dx = affine2.backward(dx);                // (100, 100)
        dx = relu2.backward(dx);                  // (100, 100)
        Tensor dp1_flat = affine1.backward(dx);   
        //展回
        Tensor dp1 = dp1_flat.reshape({batch_size, filter_num, pool_output_size, pool_output_size});
        Tensor dz1 = pool1.backward(dp1);//还原tidu
        Tensor da1 = relu1.backward(dz1);         // (100, 30, 24, 24)
        dx = conv1.backward(da1);                 //col2im

        optimizer.update(conv1.W, conv1.dW);
        optimizer.update(conv1.b, conv1.db);
        optimizer.update(affine1.W, affine1.dW);
        optimizer.update(affine1.b, affine1.db);
        optimizer.update(affine2.W, affine2.dW);
        optimizer.update(affine2.b, affine2.db);

        if (i % 50 == 0) {
            cout << "Iteration: " << setw(4) << i 
                 << " | Loss: " << fixed << setprecision(4) << loss << endl;
        }
    }
    cout<<"done"<<endl;  
    
//test
    int test_samples=1000;
    Tensor x_test_batch({test_samples,1,28,28});
    Tensor t_test_batch({test_samples,output_size});

    for (int k = 0; k < test_samples; ++k) {
        copy(x_test.data.begin() + k * 784, x_test.data.begin() + (k + 1) * 784, x_test_batch.data.begin() + k * 784);
        copy(t_test.data.begin() + k * output_size, t_test.data.begin() + (k + 1) * output_size, t_test_batch.data.begin() + k * output_size);
    }
    Tensor a1 = conv1.forward(x_test_batch);       
    Tensor z1 = relu1.forward(a1);            
    Tensor p1 = pool1.forward(z1);            
    Tensor p1_flat = p1.reshape({test_samples, affine1_input_size}); 
    Tensor a2 = affine1.forward(p1_flat);     
    Tensor z2 = relu2.forward(a2);            
    Tensor a3 = affine2.forward(z2);

    float acc = Metrics::accuracy(a3, t_test_batch);
    cout << "\nAcc: " << fixed << setprecision(2) << acc * 100.0f << endl;


    //保存模型
    conv1.W.save("W1.bin");
    conv1.b.save("b1.bin");
    affine1.W.save("W2.bin");
    affine1.b.save("b2.bin");
    affine2.W.save("W3.bin");
    affine2.b.save("b3.bin");
return 0;
 
}


//    string image_path="/home/wang/C_cnn/data/train-images-idx3-ubyte";
//     string label_path="/home/wang/C_cnn/data/train-labels-idx1-ubyte";

//     MNISTData train_dataset;
//     train_dataset.images = read_mnist_images(image_path);
//     train_dataset.labels = read_mnist_labels(label_path);
    
//     //归一化和onehot编码
//     Tensor x_train =normalize_images(train_dataset.images);
//     Tensor t_train =one_hot_encode(train_dataset.labels);
//     int train_size=x_train.shape[0];//60000

//     //初始化网罗参数
//     Tensor W1=Tensor::randn({input_size,hidden_size},0.0f,0.1f);
//     Tensor b1({hidden_size},0.0f);


//     Tensor W2 = Tensor::randn({hidden_size, output_size}, 0.0f, 0.01f);
//     Tensor b2({output_size}, 0.0f);

//     //填充逻辑
//     Affine affine1(W1, b1);
//     Relu relu;
//     Affine affine2(W2, b2);
//     SoftmaxWithLoss last_layer;
    
//     SGD optimizer(learning_rate);

//     //随机种子
//     random_device rd;
//     mt19937 gen(rd());
//     uniform_int_distribution<> dis(0, train_size - 1);


//     //训练train
//     for(int i=0;i<iters_num;++i){
//         //get minibatch
//         Tensor x_batch({batch_size,input_size});
//         Tensor t_batch({batch_size,output_size});


//         for(int k=0;k<batch_size;++k){
//             int random_idx=dis(gen);
//             //copy input data111
//             copy(x_train.data.begin() + random_idx * input_size, 
//                  x_train.data.begin() + (random_idx + 1) * input_size, 
//                  x_batch.data.begin() + k * input_size);
//             //and copy label data
//             copy(t_train.data.begin() + random_idx * output_size, 
//                  t_train.data.begin() + (random_idx + 1) * output_size, 
//                  t_batch.data.begin() + k * output_size);
//         }
//         //end 正向传播
//         Tensor a1 = affine1.forward(x_batch);
//         Tensor z1 = relu.forward(a1);
//         Tensor a2 = affine2.forward(z1);
//         float loss = last_layer.forward(a2, t_batch);

//         //and 反向传播
//         float dout = 1.0f;
//         Tensor dx = last_layer.backward(dout);
//         dx = affine2.backward(dx);
//         dx = relu.backward(dx);
//         dx = affine1.backward(dx);

//         //finally updata 权重
//         optimizer.update(affine1.W,affine1.dW);
//         optimizer.update(affine1.b,affine1.db);
//         optimizer.update(affine2.W,affine2.dW);
//         optimizer.update(affine2.b,affine2.db);

//         if (i % 500 == 0) {
//             cout << "Iteration: " << setw(5) << i 
//                  << " | 当前 Batch 误差 (Loss): " << fixed << setprecision(4) << loss << endl;
//         }
//     }

//     cout<<"训练完成"<<endl;

//     //测试集
//     string test_image_path = "/home/wang/C_cnn/data/t10k-images-idx3-ubyte"; 
//     string test_label_path = "/home/wang/C_cnn/data/t10k-labels-idx1-ubyte";

//     MNISTData test_dataset;
//     test_dataset.images = read_mnist_images(test_image_path);
//     test_dataset.labels = read_mnist_labels(test_label_path);

//     //归一化
//     Tensor x_test = normalize_images(test_dataset.images);
//     int test_size = x_test.shape[0]; // 应该刚好是 10000

//     int correct_cnt = 0;     // 记录预测正确的数量
//     int test_batch_size = 100; // 一次测100

//     for(int i = 0; i < test_size; i += test_batch_size){
//         Tensor x_batch({test_batch_size, input_size});
//         //copy
//         copy(x_test.data.begin() + i * input_size, 
//              x_test.data.begin() + (i + test_batch_size) * input_size, 
//              x_batch.data.begin());

//         //预测 前向传播
//         Tensor a1 = affine1.forward(x_batch);
//         Tensor z1 = relu.forward(a1);
//         Tensor y = affine2.forward(z1); // y 是最后一层的原始得分 (Batch_size, 10)


//         for(int k = 0; k < test_batch_size; k++) 
//             {
//             int pred_label = 0;
//             float max_score = y.data[k * output_size];
//             for(int j = 1; j < output_size; j++) {
//                 if(y.data[k * output_size + j] > max_score) {
//                     max_score = y.data[k * output_size + j];
//                     pred_label = j; // 记录得分最高的索引 (0~9)
//                 }
//             }
            
//             // 如果预测标签等于真实标签，答对题数 +1
//             if(pred_label == test_dataset.labels[i + k]) {
//                 correct_cnt++;
//             }
//             }
//         }
//         float accuracy = (float)correct_cnt / test_size;;
//         cout << "测试集总样本数: " << test_size <<endl;
//         cout << "预测正确: " << correct_cnt <<endl;
//         cout << "Accuracy: " << fixed << setprecision(2) << accuracy * 100.0f << "%\n";
//         int random_test_idx = 42;
//         cout << "\n抽取测试集第 " << random_test_idx << " 张图像：\n";
//         print_image_ascii(test_dataset.images[random_test_idx], test_dataset.labels[random_test_idx]);
        
//         Tensor single_x({1, input_size});
//         copy(x_test.data.begin() + random_test_idx * input_size, 
//             x_test.data.begin() + (random_test_idx + 1) * input_size, 
//             single_x.data.begin());
        
//         Tensor a1_s = affine1.forward(single_x);
//         Tensor z1_s = relu.forward(a1_s);
//         Tensor y_s = affine2.forward(z1_s);
        
//         int pred_label = 0; float max_score = y_s.data[0];
//         for(int j=1; j<10; j++) { if(y_s.data[j] > max_score) { max_score = y_s.data[j]; pred_label = j; } }
        
//         cout << "预测结果是: [" << pred_label << "]\n";
    

    
//     return 0;
