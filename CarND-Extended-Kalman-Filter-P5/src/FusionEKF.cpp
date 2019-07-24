#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

//-----------------------------------------------------------------------------
/**
 * Constructor.
 */
FusionEKF::FusionEKF()
{
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  /**
   * Finish initializing the FusionEKF.
   * Set the process and measurement noises
   */
  // Lidar - measurement matrix
  H_laser_ = MatrixXd(2, 4);
  H_laser_	<<	1, 0, 0, 0,
				0, 1, 0, 0;

  // Radar - jacobian matrix
  Hj_ = MatrixXd(3, 4);
  Hj_		<<	0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0;

  // Initialize ekf state
  ekf_.x_ = VectorXd(4); // state is [px, py, vx, vy]. intialise at first measurement
  ekf_.x_ << 1, 1, 1, 1;


  // Initialize state covariance matrix P
  // On first measurement we know position with certainty, velocity is uncertain
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_	<<	1,	0,	0,	 0,
				0,	1,	0,	 0,
				0,	0, 1000, 0,
				0,	0, 0,	1000;

  // Initial transition matrix F_
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ <<	1, 0, 1, 0,
              0, 1, 0, 1,
              0, 0, 1, 0,
              0, 0, 0, 1;

  // Initialize process noise covariance matrix
  ekf_.Q_ = MatrixXd::Zero(4,4); // will set non-zero elements at measurement


  // noise
  noise_ax_ = 9;
  noise_ay_ = 9;
}

//-----------------------------------------------------------------------------
/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

//-----------------------------------------------------------------------------
void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack)
{
  /**
   * Initialization
   */
  if (!is_initialized_)
  {
    /**
     * Initialize the state ekf_.x_ with the first measurement.
     * Create the covariance matrix.
     * You'll need to convert radar from polar to cartesian coordinates.
     */

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
    {
      // Extract values from measurement
      const float rho		= measurement_pack.raw_measurements_(0);
      const float phi		= measurement_pack.raw_measurements_(1);
      const float rho_dot	= measurement_pack.raw_measurements_(2);

      // Convert from polar to cartesian coordinates
      float px = rho * cos(phi);
      float py = rho * sin(phi);
      float vx = rho_dot * cos(phi);
      float vy = rho_dot * sin(phi);

      // Initialize state
      ekf_.x_ << px, py, vx, vy;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER)
    {
      // Extract values from measurement
      float px = measurement_pack.raw_measurements_(0);
      float py = measurement_pack.raw_measurements_(1);

      // Initialize state
      ekf_.x_ << px, py, 0.0, 0.0;
    }

    // done initializing, no need to predict or update
    previous_timestamp_ = measurement_pack.timestamp_;
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   * - Update the state transition matrix F according to the new elapsed time.
   *   Time is measured in seconds.
   * - Update the process noise covariance matrix.
   *   Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  const float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;

  //1. Modify the state transition matrix F according to elapsed time dt
  ekf_.F_(0,2) = dt;
  ekf_.F_(1,3) = dt;

  //2. Set the process covariance matrix Q
  const float dt2 = pow(dt,2)/2.;
  const float dt3 = pow(dt,3)/2.;

  ekf_.Q_(0,0) = dt2*dt2*noise_ax_;
  ekf_.Q_(0,2) = dt3*noise_ax_;
  ekf_.Q_(1,1) = dt2*dt2*noise_ay_;
  ekf_.Q_(1,3) = dt3*noise_ay_;
  ekf_.Q_(2,0) = ekf_.Q_(0,2);
  ekf_.Q_(2,2) = 2*dt2*noise_ax_;
  ekf_.Q_(3,1) = ekf_.Q_(1,3);
  ekf_.Q_(3,3) = 2*dt2*noise_ay_;

  ekf_.Predict();

  /**
   * Update
   * - Use the sensor type to perform the update step.
   * - Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
  {
    // Radar updates
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  }
  else 
  {
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}

//-----------------------------------------------------------------------------