#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if ((estimations.size() != ground_truth.size()) || (estimations.size() == 0)) {
    throw std::invalid_argument("The estimation vector size should not be zero and equal ground truth vector size");
  }

  //accumulate squared residuals
  for (int i = 0; i < estimations.size(); ++i) {
    VectorXd diff = estimations[i] - ground_truth[i];
    diff = diff.array()*diff.array();
    rmse += diff;
  }

  //calculate the mean
  rmse = rmse.array() / estimations.size();
  //calculate the squared root
  rmse = rmse.array().sqrt();

  return rmse;
}
