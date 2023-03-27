#include <iostream>
#include <fstream>
#include <strstream>
#include <vector>
#include <tuple>

#include "../../eigen/Eigen/Dense"

using namespace std;


Eigen::VectorXf use_func_on_values(float (*Activation_function)(float), Eigen::VectorXf* v)
{
    Eigen::VectorXf ret = *v;
    for (int i = 0; i < ret.size(); i++)
    {
        ret[i] = Activation_function(ret[i]);
    }
    return ret;
}


vector<tuple<int, Eigen::VectorXf>> read_data(string file_name);

void print_image(tuple<int, Eigen::VectorXf>& image);

float ReLU(float x);
float D_ReLU(float x);

float LeakyReLU(float x);
float D_LeakyReLU(float x);


class Node
{
public:
    Node() {}

    Node(float* Value, float* bias, float* z, Eigen::MatrixXf* Weights, int index, Eigen::VectorXf* Father_Layer, float (*Activation_function)(float), float (*Derivative_Function)(float))
    {
        _z = z;
        _value = Value;
        _bias = bias;
        _father_layer = Father_Layer;
        _weights = Weights;
        _index = index;
        _f = Activation_function;
        _d = Derivative_Function;
    }

    ~Node() {}

    void calculate_z()
    {
        *_z = _weights->row(_index).dot(*_father_layer) + *_bias;
    }

    void activate()
    {
        calculate_z();
        *_value = _f(*_z);
    }

    

    float get_value() {return *_value;}
    float get_bias() { return *_bias; }
    float get_z() { return *_z; }
    Eigen::VectorXf get_weights() { return *_weights; }
    Eigen::VectorXf get_father_layer() { return *_father_layer; }
    function<float(float)> get_activation_function() { return *_f; }
    function<float(float)> get_derivative_function() { return *_d; }

    void set_value(float x) { *_value = x; }

private:
    float* _value;
    float* _z;
    float* _bias;
    Eigen::MatrixXf* _weights;
    int _index;
    Eigen::VectorXf* _father_layer;
    float (*_f)(float);
    float (*_d)(float);

};

class Deep_Net 
{
public:
    Deep_Net() {}

    Deep_Net(vector<float> size, float (*Activation_function)(float), float (*Derivative_Function)(float))
    {
        Eigen::VectorXf temp_vec;

        _nodes.resize(size.size());
        _values.resize(size.size());
        _zs.resize(size.size());
        _biases.resize(size.size());
        _weights.resize(size.size());

        _value_changes.resize(size.size());
        _z_changes.resize(size.size());
        _weight_changes.resize(size.size());

        for (int i = 0; i < size.size(); i++)
        {
            _nodes[i].resize(size[i]);
            _values[i].resize(size[i]);
            _zs[i].resize(size[i]);
            _biases[i].resize(size[i]);

            _value_changes[i].resize(size[i]);
            _z_changes[i].resize(size[i]);

            for (int j = 0; j < _nodes[i].size(); j++)
            {
                _values[i][j] = 0;
                _zs[i][j] = 0;
                _biases[i][j] = 0;

                _value_changes[i][j] = 0;
                _z_changes[i][j] = 0;
                if (i > 0)
                {
                    _weights[i].resize(_nodes[i].size(), _nodes[i - 1].size());
                    _weight_changes[i].resize(_nodes[i].size(), _nodes[i - 1].size());

                    for (int k = 0; k < _weights[i].cols(); k++)
                    {
                        _weights[i](j, k) = 1;
                        _weight_changes[i](j, k) = 0;

                    }
                    _nodes[i][j] = Node(&_values[i][j], &_biases[i][j], &_zs[i][j], &_weights[i], j, &_values[i - 1], Activation_function, Derivative_Function);
                }
                else
                {
                    _nodes[i][j] = Node(&_values[i][j], &_biases[i][j], &_zs[i][j], &_weights[i], j, &temp_vec, Activation_function, Derivative_Function);
                }
            }
        }


        _value_changes = _values;
        _z_changes = _zs;
        _weight_changes = _weights;
    }

    ~Deep_Net() {}

    void set_input(Eigen::VectorXf v)
    {
        _values[0] = v;
    }

    void propogate()
    {
        for (int i = 1; i < _nodes.size(); i++)
        {
            for (int j = 0; j < _nodes[i].size(); j++)
            {
                _nodes[i][j].activate();
            }
        }
    }

    float calculate_err(Eigen::VectorXf exp_out)
    {
        Eigen::VectorXf ret = _values[_values.size() - 1] - exp_out;
        return ret.squaredNorm();
    }

    void calculate_changes(Eigen::VectorXf exp_out)
    {
        _value_changes[_value_changes.size() - 1] += 2 * (_values[_values.size() - 1] - exp_out);

        for (int i = 0; i < _z_changes[_z_changes.size() - 1].size(); i++)
        {
            _z_changes[_z_changes.size() - 1][i] += _nodes[_nodes.size() - 1][i].get_derivative_function()(_zs[_zs.size() - 1][i]);
        }

        for (int i = 0; i < _weight_changes[_weight_changes.size() - 1].rows(); i++)
        {
            auto Fl = _nodes[_nodes.size() - 1][i].get_father_layer();
            auto zc = _z_changes[_z_changes.size() - 1][i];
            for (int j = 0; j < _weight_changes[_weight_changes.size() - 1].cols(); j++)
            {
                _weight_changes[_weight_changes.size() - 1](i, j) += Fl[j] * zc;
            }
        }



        for (int i = _value_changes.size() - 2; i > 0; i--)
        {
            for (int j = 0; j < _value_changes[i].size(); j++)
            {
                _value_changes[i][j] += _weights[i + 1].col(j).dot(_z_changes[i+1]);
            }

            for (int j = 0; j < _z_changes[i].size(); j++)
            {
                _z_changes[i][j] += _nodes[i][j].get_derivative_function()(_zs[i][j]);
            }

            for (int j = 0; j < _weight_changes[_weight_changes.size() - 1].rows(); j++)
            {
                auto Fl = _nodes[i][j].get_father_layer();
                auto zc = _z_changes[i][j];
                for (int k = 0; k < _weight_changes[i].cols(); k++)
                {
                    _weight_changes[i](j, k) += Fl[k] * zc;
                }
            }
        }
    }

    void apply_changes()
    {
        for (int i = 0; i < _nodes.size(); i++)
        {
            _values[i] += _value_changes[i];
            _biases[i] += _z_changes[i];
            _weights[i] += _weight_changes[i];
        }
    }

    Eigen::VectorXf get_output() { return _values[_values.size()-1]; }

    vector<vector<Node>> get_nodes() { return _nodes; }
    vector<Eigen::VectorXf> get_values() { return _values; }


private:
    vector<vector<Node>> _nodes;
    vector<Eigen::VectorXf> _values;
    vector<Eigen::VectorXf> _zs;
    vector<Eigen::VectorXf> _biases;
    vector<Eigen::MatrixXf> _weights;



    vector<Eigen::VectorXf> _value_changes;
    vector<Eigen::VectorXf> _z_changes;
    vector<Eigen::MatrixXf> _weight_changes;

};



int main()
{

    /*vector<tuple<int, Eigen::VectorXf>> train_data = read_data(".\\..\\MNIST_data\\mnist_train.txt");
    vector<tuple<int, Eigen::VectorXf>> test_data = read_data(".\\..\\MNIST_data\\mnist_test.txt");

    print_image(test_data[100]);*/
    
    
    /*Eigen::VectorXf W;
    Eigen::VectorXf FL;

    W.resize(10);
    FL.resize(10);
    for (int i = 0; i < 10; i++)
    {
        W(i) = i;
        FL(i) = i;
    }

    float Nf = float(0);

    Node N(&Nf, &FL, &LeakyReLU, &D_LeakyReLU);

    cout << W << endl;

    cout << W.dot(FL) << endl;

    cout << N.get_weights()(3) << " " << N.get_father_layer()(3)  << endl;

    cout << N.get_value() << endl;
    N.activate();
    cout << N.get_value() << endl;*/


    vector<tuple<int, Eigen::VectorXf>> test_data = read_data(".\\..\\MNIST_data\\mnist_test.txt");

    Deep_Net Net(vector<float>({ 784, 16, 16, 10 }), &LeakyReLU, &D_LeakyReLU);

    Net.set_input(get<1>(test_data[100]));

    cout << Net.get_output() << endl << endl;
    Net.propogate();
    cout << Net.get_output() << endl << endl;

    Eigen::VectorXf EXP_OUT;
    EXP_OUT.resize(10);
    for (int i = 0; i < EXP_OUT.size(); i++)
    {
        if (i == 4) EXP_OUT(i) = 1;
        else EXP_OUT(i) = 0;
    }

    cout << Net.calculate_err(EXP_OUT) << endl << endl;

    Net.calculate_changes(EXP_OUT);
    Net.apply_changes();

    Net.propogate();
    cout << Net.get_output() << endl << endl;
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
float D_ReLU(float x)
{
    return x < 0 ? float(0) : float(1);
}

float LeakyReLU(float x) 
{
    return x > 0 ? x : 0.1*x;
}
float D_LeakyReLU(float x)
{
    return x > 0 ? float(1) : float(0.1);
}

