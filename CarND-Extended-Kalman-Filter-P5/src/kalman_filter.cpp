#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

//-----------------------------------------------------------------------------
KalmanFilter::KalmanFilter() {}

//-----------------------------------------------------------------------------
KalmanFilter::~KalmanFilter() {}

//-----------------------------------------------------------------------------
void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

//-----------------------------------------------------------------------------
void KalmanFilter::Predict()
{
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

//-----------------------------------------------------------------------------
void KalmanFilter::UpdateRoutine(const Eigen::VectorXd &y)
{
  const MatrixXd Ht = H_.transpose();
  const MatrixXd S = H_ * P_ * Ht + R_;
  const MatrixXd Si = S.inverse();

  // compute Kalamn gain
  const MatrixXd K =  P_ * Ht * Si;

  // update estimate for new state
  x_ = x_ + (K * y);
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * H_) * P_;
}

//-----------------------------------------------------------------------------
void KalmanFilter::Update(const VectorXd &z) {
  /**
   * update the state by using Kalman Filter equations
   */
  VectorXd y = z - H_ * x_;
  UpdateRoutine(y);
}

//-----------------------------------------------------------------------------
void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * update the state by using Extended Kalman Filter equations
   */
	// recover explicitly status information
	const float px = x_(0);
	const float py = x_(1);
	const float vx = x_(2);
	const float vy = x_(3);

  // map predicted state from cartesian to polar coordinates
	double rho     = sqrt(px * px + py * py);
	double phi	   = atan2(py, px);
	double rho_dot = (px * vx + py * vy) / std::max(rho, 0.0001);

	VectorXd h_x(3);
	h_x << rho, phi, rho_dot;

  VectorXd y = z - h_x;

  // normalize phi
  while (y(1) > M_PI)
  {
    y(1) -= 2 * M_PI;
  }
	while (y(1) < -M_PI)
  {
    y(1) += 2 * M_PI;
  }
  
  UpdateRoutine(y);
}

//-----------------------------------------------------------------------------