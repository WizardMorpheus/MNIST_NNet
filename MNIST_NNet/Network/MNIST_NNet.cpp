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
    fstream fin;

    // Open an existing file
    fin.open(file_name, ios::in);

    // Read the Data from the file
    // as String Vector
    string line, word, temp;

    while (fin >> temp) {

        Eigen::VectorXf row;

        // read an entire row and
        // store it in a string variable 'line'
        getline(fin, line);

        // used for breaking words
        stringstream s(line);

        // read every column data of a row and
        // store it in a string variable, 'word'

        getline(s, word, ',');

        int label = stoi(word);

        while (getline(s, word, ',')) {

            // add all the column data
            // of a row to a vector
            row.resize(row.size() + 1);
            row(row.size() - 1) = stof(word);
        }
        ret.push_back(make_tuple(label, row));
    }
}


int main()
{
	Eigen::Vector3f v1(1, 0, 0);
	Eigen::Vector3f v2(0, 1, 0);

	Eigen::Vector3f v3 = v1.cross(v2);

	std::cout << v1 << std::endl << std::endl << v2 << std::endl << std::endl << v3;

    vector<tuple<int, Eigen::VectorXf>> train_data = read_data("../../MNIST_Data/mnist_train.csv");
    vector<tuple<int, Eigen::VectorXf>> test_data = read_data("../../MNIST_Data/mnist_test.csv");

}