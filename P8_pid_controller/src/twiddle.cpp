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
*/

#include "twiddle.h"

#include <assert.h>


double sum(const vector<double>& vec) {
  double result = 0;
  for (auto& v: vec) {
    result += v;
  }

  return result;
}


Twiddle::Twiddle(
  const vector<double>& params,
  const vector<double>& dparams,
  const double& threshold) {

  size_t n = params.size();
  for (size_t i = 0; i < n; i++) {
    this->p.push_back(params[i]);
    this->dp.push_back(dparams[i]);
  }

  this->pni = false;
  this->num_params = n;
  this->best_err_set = false;
  this->threshold = threshold;
  this->param_idx = -1;
}


void Twiddle::UpdateNext() {
  this->param_idx = (this->param_idx + 1) % this->num_params;
  this->p[this->param_idx] += this->dp[this->param_idx];
}


void Twiddle::Update(const double& err) {

  if (this->best_err_set == false) {
    this->best_err = err;
    this->best_err_set = true;
    this->UpdateNext();
    return;
  }

  if (sum(this->dp) <= this->threshold) {
    return;
  }

  assert(this->param_idx >= 0);
  size_t i = this->param_idx;
  if (err < this->best_err) {
    this->best_err = err;
    if (this->pni == true) {
      // Previous error check resulted in no improvement
      this->dp[i] *= 1.05;
      this->pni = false;
    } else {
      this->dp[i] *= 1.1;
    }
    this->UpdateNext();

  } else {
    // No improvement in `best_err`
    if (this->pni == true) {
      this->p[i] += this->dp[i];
      this->dp[i] *= 0.95;
      this->pni = false;
      this->UpdateNext();
    } else {
      this->p[i] -= 2*this->dp[i];
      // Set no improvement flag for param i
      this->pni = true;
    }
  }
}
