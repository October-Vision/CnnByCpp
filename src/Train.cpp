#include "Train.hpp"

using namespace std;


Trainer::Trainer(const CNNConfig& config):cfg(config) {}


void Trainer::fit(SimpleConvNet& model,
    const string& train_img_path, const string& train_lbl_path,
    const string& test_img_path,  const string& test_lbl_path)
{
    cout<<"6667开始训练_kskbl_zdjd_77888"<<endl;
    MNISTData train_dataset, test_dataset;
    train_dataset.images = read_mnist_images(train_img_path);
    train_dataset.labels = read_mnist_labels(train_lbl_path);
    test_dataset.images  = read_mnist_images(test_img_path);
    test_dataset.labels  = read_mnist_labels(test_lbl_path);

    //归一化和one-hot编码
    Tensor x_train=normalize_images(train_dataset.images).reshape({60000, 1, 28, 28});
    Tensor t_train=one_hot_encode(train_dataset.labels, cfg.output_size);
    Tensor x_test=normalize_images(test_dataset.images).reshape({10000, 1, 28, 28});
    Tensor t_test=one_hot_encode(test_dataset.labels, cfg.output_size);
    //优化器引入
    int train_size=x_train.shape[0];
    SGD optimizer(cfg.learning_rate);

    //随机数
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, train_size-1);

    for (int i=0; i<cfg.iters_num; ++i) {
        Tensor x_batch({cfg.batch_size, 1, 28, 28}); 
        Tensor t_batch({cfg.batch_size, cfg.output_size}); 
        
        for (int k = 0; k < cfg.batch_size; ++k) {
            int random_idx = dis(gen); 
            copy(x_train.data.begin() + random_idx * 784, x_train.data.begin() + (random_idx + 1) * 784, x_batch.data.begin() + k * 784);
            copy(t_train.data.begin() + random_idx * cfg.output_size, t_train.data.begin() + (random_idx + 1) * cfg.output_size, t_batch.data.begin() + k * cfg.output_size);
        }

        float loss = model.forward_loss(x_batch, t_batch);
        model.backward();
        model.update_weights(optimizer);

        if (i%50 == 0) {
            cout << "      Iteration: " << setw(4) << i 
                 << " | Loss: " << fixed << setprecision(4) << loss << endl;
        }
    }

//test
    int test_samples = 1000; 
    Tensor x_test_batch({test_samples, 1, 28, 28});
    Tensor t_test_batch({test_samples, cfg.output_size});
    
    for (int k = 0; k < test_samples; ++k) {
        copy(x_test.data.begin()+k*784, x_test.data.begin() +(k+1)*784, x_test_batch.data.begin()+k*784);
        copy(t_test.data.begin()+k*cfg.output_size, t_test.data.begin() +(k + 1) * cfg.output_size, t_test_batch.data.begin() + k*cfg.output_size);
    }
    Tensor test_pred=model.predict(x_test_batch);
    float acc=Metrics::accuracy(test_pred, t_test_batch);
    cout << "Acc:" << fixed << setprecision(2) << acc*100.0f<< endl;
    model.save_weights();

}
