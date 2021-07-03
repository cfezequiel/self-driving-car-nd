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

#ifndef TWIDDLE_H
#define TWIDDLE_H

#include <vector>

using std::size_t;
using std::vector;


// Implements twiddle algorithm
class Twiddle {
  public:
    Twiddle(
        const vector<double>& params,
        const vector<double>& dparams,
        const double& threshold=0.001);

    void UpdateNext();
    void Update(const double& err);

    vector<double> p;
    vector<double> dp;

    // "Previous no improvement" (PNI)
    // Per param flag indicating whether previous param change led to an
    // improvement in the error or not.
    bool pni;
    size_t num_params;
    int param_idx;
    bool best_err_set;
    double best_err;
    double threshold;
};

double sum(const vector<double>& vec);

#endif //TWIDDLE_H
