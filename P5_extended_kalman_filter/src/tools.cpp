/*
 Copyright 2020 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

 This file incorporates work covered by the following copyright and
 permission notice:

 MIT License

 Copyright (c) 2016-2019 Udacity, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
*/

#include "tools.h"
#include <iostream>
#include <math.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;
using std::endl;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

  VectorXd rmse(4);
  rmse << 0,0,0,0;

  // Checks the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if (estimations.size() != ground_truth.size()
      || estimations.size() == 0) {
    cout << "Invalid estimation or ground_truth data." << endl;
    return rmse;
  }

  // Accumulate squared residuals
  for (unsigned int i=0; i < estimations.size(); ++i) {

    VectorXd residual = estimations[i] - ground_truth[i];

    // Coefficient-wise multiplication
    residual = residual.array()*residual.array();
    rmse += residual;
  }

  // Calculate the mean
  rmse = rmse/estimations.size();

  // calculate the squared root
  rmse = rmse.array().sqrt();

  // return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

  // Recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  MatrixXd Hj(3,4);

  // Compute the Jacobian matrix
  float c1 = px*px + py*py;
  if (fabs(c1) < 0.0001) {
    cout << "Warning: px and py are both zero." << endl;
    Hj << MatrixXd::Zero(3, 4);
  	return Hj;
  }
  float c2 = sqrt(c1);
  float c3 = c2*c2*c2;
  Hj << px/c2, py/c2, 0, 0,
        -py/c1, px/c1, 0, 0,
        py*(vx*py - vy*px)/c3, px*(vy*px - vx*py)/c3, px/c2, py/c2;

  return Hj;
}


VectorXd Tools::Polar2Cartesian(const Eigen::VectorXd& polar) {
  float rho = polar[0];
  float phi = polar[1];

  VectorXd x(4);
  x(0) = rho*cos(phi);  // px
  x(1) = rho*sin(phi);  // py

  return x;
}


VectorXd Tools::Cartesian2Polar(const Eigen::VectorXd& cart) {
  float px = cart[0];
  float py = cart[1];
  float vx = cart[2];
  float vy = cart[3];

  float rho = sqrt(px*px + py*py);
  float phi = atan2(py, px);
  float rho_dot;
  if (fabs(rho) < 0.0001) {
  	rho_dot = 0;
  } else {
	rho_dot = (px*vx + py*vy) / rho;
  }
  VectorXd x(3);
  x << rho, phi, rho_dot;
  return x;
}


float Tools::NormalizeAngle(const float &angle_rad) {
  float rho = angle_rad;
  if (rho < -M_PI) {
    do {
      rho += M_PI;
    } while (rho < -M_PI);
  }
  else if (rho > M_PI) {
    do {
      rho -= M_PI;
    } while (rho > M_PI);
  }
  return rho;
}