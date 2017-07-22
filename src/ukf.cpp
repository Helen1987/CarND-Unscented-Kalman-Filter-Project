#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <cmath>

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
  x_ << 0, 0, 0, 0, 0;

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.6;

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

  is_initialized_ = false;

  NIS_laser_ = 0;
  NIS_radar_ = 0;

  n_x_ = 5;
  n_aug_ = 7;
  n_sigma = 2 * n_aug_ + 1;

  //define spreading parameter
  lambda_ = 3 - n_aug_;

  //set weights
  weights_ = VectorXd(n_sigma);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  auto rest_weights = 0.5 / (lambda_ + n_aug_);
  for (int i = 1; i < n_sigma; ++i) {
    weights_(i) = rest_weights;
  }

  R_laser_ = MatrixXd(2, 2);
  R_laser_ << std_laspx_*std_laspx_, 0,
              0, std_laspy_*std_laspy_;

  R_radar_ = MatrixXd(3, 3);
  R_radar_ << std_radr_*std_radr_, 0, 0,
              0, std_radphi_*std_radphi_, 0,
              0, 0, std_radrd_*std_radrd_;

  H_laser_ = MatrixXd(2, n_x_);
  H_laser_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0;
}

UKF::~UKF() {}

double UKF::NormalizeAngle(double angle) const {
  auto result = fmod(angle + M_PI, 2 * M_PI) - M_PI;
  return std::abs(result) > negligible ? result : 0;
}

MatrixXd UKF::GenerateSigmaPoints() {
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  // fill augmented mean state
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
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sigma);
  Xsig_aug.col(0) = x_aug;
  auto coeff = sqrt(lambda_ + n_aug_);
  for (int i = 0; i < n_aug_; i++)
  {
    auto variance = coeff * L.col(i);
    Xsig_aug.col(i + 1) = x_aug + variance;
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - variance;
  }
  return Xsig_aug;
}

void UKF::PredictSigmaPoints(double delta_t) {
  MatrixXd Xsig_aug = GenerateSigmaPoints();

  double p_x, p_y, v, psi, psi_dot, a, yawdd;
  VectorXd predicted_sigma = VectorXd(n_x_);

  Xsig_pred_ = MatrixXd(n_x_, n_sigma);
  double delta_t2 = delta_t*delta_t;

  for (int i = 0; i < n_sigma; ++i) {
    p_x = Xsig_aug(0, i);
    p_y = Xsig_aug(1, i);
    v = Xsig_aug(2, i);
    psi = Xsig_aug(3, i);
    psi_dot = Xsig_aug(4, i);
    a = Xsig_aug(5, i);
    yawdd = Xsig_aug(6, i);
    //predict sigma points
    if (std::abs(psi_dot) > negligible) {
      predicted_sigma(0) = p_x + v*(sin(psi + psi_dot*delta_t) - sin(psi)) / psi_dot
        + delta_t2*cos(psi)*a / 2;
      predicted_sigma(1) = p_y + v*(-cos(psi + psi_dot*delta_t) + cos(psi)) / psi_dot
        + delta_t2*sin(psi)*a / 2;
      predicted_sigma(2) = v + delta_t*a;
      predicted_sigma(3) = NormalizeAngle(psi + psi_dot*delta_t + delta_t2*yawdd / 2);
      predicted_sigma(4) = psi_dot + delta_t*yawdd;
    }
    else { //avoid division by zero
      predicted_sigma(0) = p_x + v*cos(psi)*delta_t + delta_t2*cos(psi)*a / 2;
      predicted_sigma(1) = p_y + v*sin(psi)*delta_t + delta_t2*sin(psi)*a / 2;
      predicted_sigma(2) = v + delta_t*a;
      predicted_sigma(3) = NormalizeAngle(psi + psi_dot*delta_t + delta_t2*a / 2);
      predicted_sigma(4) = psi_dot + delta_t*yawdd;
    }
    //write predicted sigma points into right column
    Xsig_pred_.col(i) = predicted_sigma;
  }
}

void UKF::UpdateState(const VectorXd &z, const MatrixXd &Zsig) {
  //mean predicted measurement
  VectorXd z_pred = VectorXd(z.size());
  z_pred.fill(0.0);
  //calculate mean predicted measurement
  for (int i = 0; i < n_sigma; ++i) {
    z_pred += weights_(i)*Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(z.size(), z.size());
  S.fill(0.0);
  //calculate measurement covariance matrix S
  for (int i = 0; i < n_sigma; ++i) {
    VectorXd diff = Zsig.col(i) - z_pred;
    diff(1) = NormalizeAngle(diff(1));

    S += weights_(i)*diff*diff.transpose();
  }
  S += R_radar_;

  VectorXd z_diff, x_diff;
  MatrixXd Tc = MatrixXd(n_x_, z.size());

  Tc.fill(0.0);
  //calculate cross correlation matrix
  for (int i = 0; i < n_sigma; ++i) {
    z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    z_diff(1) = NormalizeAngle(z_diff(1));

    // state difference
    x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    x_diff(3) = NormalizeAngle(x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  //calculate Kalman gain K;
  MatrixXd K = Tc*S.inverse();
  //update state mean and covariance matrix
  z_diff = z - z_pred;
  z_diff(1) = NormalizeAngle(z_diff(1));

  x_ += K*z_diff;
  x_(3) = NormalizeAngle(x_(3));
  P_ -= K*S*K.transpose();
  NIS_radar_ = z_diff.transpose()*S.inverse()*z_diff;
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {
    // ignore zero-values measurements and
    // wait for sufficient measurement
    if (std::abs(pow(meas_package.raw_measurements_(0), 2) + pow(meas_package.raw_measurements_(1), 2)) < negligible)
      return;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates and initialize state.
      double ro = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      double ro_dot = meas_package.raw_measurements_(2);

      x_ << ro*cos(phi), ro*sin(phi), ro_dot, 0, 0;

      P_ << 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 5, 0,
            0, 0, 0, 0, 5;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_ << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), 0, 0, 0;

      P_ << 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 2, 0, 0,
            0, 0, 0, 2, 0,
            0, 0, 0, 0, 2;
    }

    previous_timestamp_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  if (!use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER)
    return;
  if (!use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR)
    return;

  double delta_t = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = meas_package.timestamp_;

  if (delta_t > negligible) {
    Prediction(delta_t);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    PredictSigmaPoints(delta_t);
  }


  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }
  else {
    UpdateRadar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  PredictSigmaPoints(delta_t);

  //predict state mean
  x_.fill(0.0);
  for (int i = 0; i < n_sigma; ++i) {
    x_ += weights_(i)*Xsig_pred_.col(i);
  }
  x_(3) = NormalizeAngle(x_(3));
  //predict state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < n_sigma; ++i) {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    x_diff(3) = NormalizeAngle(x_diff(3));

    P_ += weights_(i)*x_diff*x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  VectorXd y = meas_package.raw_measurements_ - H_laser_ * x_;
  MatrixXd Ht = H_laser_.transpose();
  MatrixXd PHt = P_ * Ht;
  MatrixXd S = H_laser_ * PHt + R_laser_;
  MatrixXd Si = S.inverse();
  MatrixXd K = PHt * Si;

  //new state
  x_ = x_ + (K * y);
  long x_size = x_.size();
  P_ = (MatrixXd::Identity(x_size, x_size) - K * H_laser_) * P_;

  NIS_laser_ = (y.transpose()*Si*y);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  //set measurement dimension, radar can measure r, phi, and r_dot
  int const n_z = 3;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sigma);

  double p_x, p_y, v, psi, psi_dot;
  //transform sigma points into measurement space
  for (int i = 0; i < n_sigma; ++i) {
    p_x = Xsig_pred_(0, i);
    p_y = Xsig_pred_(1, i);
    v = Xsig_pred_(2, i);
    psi = Xsig_pred_(3, i);
    psi_dot = Xsig_pred_(4, i);

    double rho = sqrt(p_x*p_x + p_y*p_y);
    Zsig(0, i) = rho;
    // phi
    Zsig(1, i) = p_x == 0 ? 0 : atan2(p_y, p_x);
    // rho_dot
    Zsig(2, i) = (p_x*cos(psi)*v + p_y*sin(psi)*v) / (std::abs(rho) > negligible ? rho : negligible);
  }

  UpdateState(meas_package.raw_measurements_, Zsig);
}
