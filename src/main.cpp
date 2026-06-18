#include "main.hpp"
using namespace std;
using namespace cv;

void print_ascii_image(const Tensor& img) {
    cout << "\n------------------------------\n";
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            float pixel = img.data[i * 28 + j];
            if (pixel > 0.8f) cout << "@@";
            else if (pixel > 0.5f) cout << "##";
            else if (pixel > 0.2f) cout << "..";
            else cout << "  ";
        }
        cout << "\n";
    }
    cout << "------------------------------\n";
}

int main() {
    // //gui部分手写画板
    // CNNConfig config;
    // config.filter_num = 30;
    // config.hidden_size = 100;
    // Predictor predictor(config, "../model/");

    // DrawPad pad("CNN Draw Pad", 280, 12, &predictor);
    // pad.run();
    
    // return 0;





    // /* ##############训练部分*/
    //参数配置
    CNNConfig config;
    config.iters_num = 10000; 
    config.batch_size = 100;
    config.learning_rate = 0.05f;
    //config.hidden_size = 200;

    //数据集
    std::string train_images = "/home/wang/C_cnn/data/train-images-idx3-ubyte";
    std::string train_labels = "/home/wang/C_cnn/data/train-labels-idx1-ubyte";
    std::string test_images  = "/home/wang/C_cnn/data/t10k-images-idx3-ubyte";
    std::string test_labels  = "/home/wang/C_cnn/data/t10k-labels-idx1-ubyte";

    //构建模型对象与调度器对象
    SimpleConvNet model(config);
    Trainer trainer(config);

    //训练，并自动保存
    trainer.fit(model, train_images, train_labels, test_images, test_labels);
   
   
   
   
   
    // /*#############推理部分*/
    // CNNConfig config;
    // config.filter_num = 30;
    // config.hidden_size = 100;

    // Predictor Predictor(config,"../model/");
    // string test_images = "/home/wang/C_cnn/data/t10k-images-idx3-ubyte";
    // string test_labels = "/home/wang/C_cnn/data/t10k-labels-idx1-ubyte";
    // vector<vector<uint8_t>> raw_images = read_mnist_images(test_images);
    // vector<uint8_t> raw_labels = read_mnist_labels(test_labels);
    
    // random_device rd;
    // mt19937 gen(rd());
    // uniform_int_distribution<> dis(0, 9999);

    // string cmd;
    // while (getline(cin, cmd) && cmd != "q") {
    //     // 随机抽一张图
    //     int idx = dis(gen);
    //     Mat src1(28, 28, CV_8UC1, raw_images[idx].data());
    //     imshow("origin",src1);
    //     waitKey(10);
        
    //     // 提取这张单图的数据并归一化 (重塑为 1x1x28x28 的 4D 张量)
    //     Tensor single_img({1, 1, 28, 28}, 0.0f);
    //     for (int i = 0; i < 784; ++i) {
    //         single_img.data[i] = raw_images[idx][i] / 255.0f;
    //     }

    //     // 把图像画在终端上
    //     cout << "原始图像：" << endl;
    //     print_ascii_image(single_img);

    //     int prediction = Predictor.predict(single_img);

    //     int ground_truth = raw_labels[idx];
        
    //     cout << "预测结果" << prediction << endl;
    //     cout << "真实答案" << ground_truth << endl;
        
    //     if (prediction == ground_truth) {
    //         cout << "yes" << endl;
    //     } else {
    //         cout << "no" << endl;
    //     }
    // }
    
    // return 0;


}