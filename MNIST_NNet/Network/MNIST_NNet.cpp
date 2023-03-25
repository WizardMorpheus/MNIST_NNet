#include <iostream>
#include <fstream>
#include <strstream>
#include <vector>
#include <tuple>

#include "../../eigen/Eigen/Dense"

using namespace std;

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


int main()
{

    vector<tuple<int, Eigen::VectorXf>> train_data = read_data(".\\..\\MNIST_data\\mnist_train.txt");
    vector<tuple<int, Eigen::VectorXf>> test_data = read_data(".\\..\\MNIST_data\\mnist_test.txt");

    print_image(test_data[100]);


}