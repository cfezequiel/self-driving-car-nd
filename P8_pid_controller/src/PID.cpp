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

 Copyright (c) 2016-2020 Udacity, Inc.

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

#include "PID.h"

/**
 * DONE: Complete the PID class. You may add any additional desired functions.
 */

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_) {
  /**
   * OK: Initialize PID coefficients (and errors, if needed)
   */
  this->UpdateParams(Kp_, Ki_, Kd_);
  this->i_error = 0;
  this->is_first_cte = true;
}

void PID::UpdateError(double cte) {
  if (this->is_first_cte == true) {
    this->p_error = cte;
    this->is_first_cte = false;
  }
  this->d_error = cte - this->p_error;
  this->p_error = cte;
  this->i_error += cte;
}

double PID::TotalError() {
  /**
   * OK: Calculate and return the total error
   */
  // Returns sum of all the errors
  return (
      this->Kp*this->p_error +
      this->Ki*this->i_error +
      this->Kd*this->d_error);
}

void PID::UpdateParams(double Kp_, double Ki_, double Kd_) {
  this->Kp = Kp_;
  this->Ki = Ki_;
  this->Kd = Kd_;
}
