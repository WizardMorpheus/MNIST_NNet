#include <iostream>

#include "../../eigen/Eigen/Dense"


int main()
{
	Eigen::Vector3f v1(1, 0, 0);
	Eigen::Vector3f v2(0, 1, 0);

	Eigen::Vector3f v3 = v1.cross(v2);

	std::cout << v1 << std::endl << std::endl << v2 << std::endl << std::endl << v3;
}