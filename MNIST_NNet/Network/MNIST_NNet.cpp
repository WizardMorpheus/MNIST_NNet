#include <iostream>
#include <fstream>
#include <strstream>
#include <vector>
#include <tuple>

#include "../../eigen/Eigen/Dense"

using namespace std;

vector<tuple<int, Eigen::VectorXf>> read_data(string file_name);

void print_image(tuple<int, Eigen::VectorXf>& image);

float ReLU(float x);

float LeakyReLu(float x);

class Node
{
public:
    Node() {}

    Node(float *Value, Eigen::VectorXf Weights, Eigen::VectorXf* Father_Layer, float (*Activation_function)(float))
    {
        _value = Value;
        _weights = Weights;
        _father_layer = Father_Layer;
        _f = Activation_function;
    }

    ~Node() {}

    void activate()
    {
        *_value = _f(_weights.dot(*_father_layer));
    }

    float get_value() {return *_value;}
    Eigen::VectorXf get_weights() { return _weights; }
    Eigen::VectorXf get_father_layer() { return *_father_layer; }
    function<float(float)> get_activation_function() { return *_f; }


private:
    float* _value;
    Eigen::VectorXf _weights;
    Eigen::VectorXf* _father_layer;
    float (*_f)(float);
};

class Deep_Net 
{
public:
    Deep_Net()
    {

    }

    ~Deep_Net() {}
private:

};



int main()
{

    /*vector<tuple<int, Eigen::VectorXf>> train_data = read_data(".\\..\\MNIST_data\\mnist_train.txt");
    vector<tuple<int, Eigen::VectorXf>> test_data = read_data(".\\..\\MNIST_data\\mnist_test.txt");

    print_image(test_data[100]);*/
    Eigen::VectorXf W;
    Eigen::VectorXf FL;

    W.resize(10);
    FL.resize(10);
    for (int i = 0; i < 10; i++)
    {
        W(i) = i;
        FL(i) = i;
    }

    float Nf = float(0);

    Node N(&Nf, W, &FL, &ReLU);

    cout << W << endl;

    cout << W.dot(FL) << endl;


    cout << N.get_weights()(3) << " " << N.get_father_layer()(3)  << endl;

    cout << N.get_value() << endl;
    N.activate();
    cout << N.get_value() << endl;



}



vector<tuple<int, Eigen::VectorXf>> read_data(string file_name)
{
    vector<tuple<int, Eigen::VectorXf>> ret;

    // File pointer
    ifstream fin;

    fin.open(file_name, ifstream::in);

    bool check = fin.is_open();

    // Read the Data from the file
    // as String Vector
    string line, word, temp;

    Eigen::VectorXf row;
    row.resize(784);


    // read every column data of a row and
    // store it in a string variable, 'word'

    while (getline(fin, line))
    {
        stringstream s(line);
        getline(s, word, ',');
        int label = stoi(word);


        for (int i = 0; i < 784; i++) {
            getline(s, word, ',');
            row(i) = stof(word);
        }

        ret.push_back(make_tuple(label, row));
    }

    return ret;
}

void print_image(tuple<int, Eigen::VectorXf>& image)
{
    char shades[4] = { ' ', char(177), char(178), char(219) };

    for (int i = 0; i < 4; i++) {
        cout << shades[i];
    }
    cout << endl;

    for (int i = 0; i < get<1>(image).size(); i++)
    {
        float num = (get<1>(image)(i));
        int index = floor((num / 256) * 4);

        if (i % 28 == 0) cout << endl;
        cout << shades[index];
    }
    cout << endl;
}

float ReLU(float x)
{
    return max(float(0), x);
}

float LeakyReLu(float x) 
{
    return x > 0 ? x : 0.1*x;
}
