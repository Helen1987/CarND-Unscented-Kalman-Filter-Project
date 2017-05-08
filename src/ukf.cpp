#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  n_x_ = 5;
  n_aug_ = 7;
  n_sigma = 2 * n_aug_ + 1;

  //define spreading parameter
  double lambda = 3 - n_aug_;
}

UKF::~UKF() {}

void UKF::GenerateSigmaPoints() {
  //create augmented mean state
  x_aug.head(n_x_) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5) = std_a_*std_a_;
  P_aug(6, 6) = std_yawdd_*std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i< n_sigma; i++)
  {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  double delta_t2 = delta_t*delta_t;
  Xsig_pred_ = MatrixXd(n_x_, n_sigma);
  double p_x, p_y, v, psi, psi_dot, a, yawdd;
  VectorXd predicted_sigma = VectorXd(n_x_);

  GenerateSigmaPoints();

  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    p_x = Xsig_aug.col(0, i);
    p_y = Xsig_aug.col(1, i);
    v = Xsig_aug.col(2, i);
    psi = Xsig_aug.col(3, i);
    psi_dot = Xsig_aug.col(4, i);
    a = Xsig_aug.col(5, i);
    yawdd = Xsig_aug.col(6, i);
    //predict sigma points
    if (std::abs(psi_dot) > negligible) {
      predicted_sigma(0) = p_x + v*(sin(psi + psi_dot*delta_t) - sin(psi)) / psi_dot
        + delta_t2*cos(psi)*a / 2;
      predicted_sigma(1) = p_y + v*(-cos(psi + psi_dot*delta_t) + cos(psi)) / psi_dot
        + delta_t2*sin(psi)*a / 2;
      predicted_sigma(2) = v + delta_t*a;
      predicted_sigma(3) = psi + psi_dot*delta_t + delta_t2*yawdd / 2;
      predicted_sigma(4) = psi_dot + delta_t*yawdd;
    }
    else { //avoid division by zero
      predicted_sigma(0) = p_x + v*cos(psi)*delta_t + delta_t2*cos(psi)*a / 2;
      predicted_sigma(1) = p_y + v*sin(psi)*delta_t + delta_t2*sin(psi)*a / 2;
      predicted_sigma(2) = v + delta_t*a;
      predicted_sigma(3) = psi + psi_dot*delta_t + delta_t2*a / 2;
      predicted_sigma(4) = psi_dot + delta_t*yawdd;
    }
    //write predicted sigma points into right column
    Xsig_pred_.col(i) = predicted_sigma;

    // predict the state
    // predict covariance matrix
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}
