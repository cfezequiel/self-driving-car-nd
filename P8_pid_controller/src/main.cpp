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

#include <math.h>
#include <uWS/uWS.h>
#include <iostream>
#include <assert.h>
#include <string>
#include <vector>
#include "json.hpp"
#include "PID.h"
#include "twiddle.h"

// for convenience
using nlohmann::json;
using std::string;
using std::vector;
using std::size_t;

// Uncomment below to enable PID gain tuning using the twiddle algorithm.
//#define USE_TWIDDLE 1

const double MAX_SPEED = 35;  // MPH
const double MIN_SPEED = 10;   // MPH

#ifdef USE_TWIDDLE
const size_t TWIDDLE_BATCH_SIZE = 50;
#endif

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_last_of("]");
  if (found_null != string::npos) {
    return "";
  }
  else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}

int main() {
  uWS::Hub h;

  PID pid;
  /**
   * OK: Initialize the pid variable.
   */
  // Initial params from: https://github.com/justinlee007/CarND-PID-Control-Project/blob/master/src/main.cpp
  //vector<double> params{0.15, 0.00004, 3.0}; // {Kp, Ki, Kd}

  // Best params from twiddle-based run
  vector<double> params{0.180675, 4.34295e-05, 3.0}; // {Kp, Ki, Kd}

#ifdef USE_TWIDDLE
    vector<double> dparams {0.15/10, 0.00004/10, 3.0/10};
    Twiddle twiddle(params, dparams);
    pid.Init(twiddle.p[0], twiddle.p[1], twiddle.p[2]);

    // Variables for getting the average (batch) error
    size_t count = 0;
    double total_error = 0;
#else
    pid.Init(params[0], params[1], params[2]);
    //pid.Init(params[0], 0, 0);  // P-only
    //pid.Init(0, params[1], 0);  // I-only
    //pid.Init(0, 0, params[2]);  // D-only
    //pid.Init(params[0], 0, params[2]);  // PD-only
#endif

  double speed_limiter = MAX_SPEED;

#ifdef USE_TWIDDLE
  h.onMessage(
      [&pid, &twiddle, &speed_limiter, &count, &total_error](
           uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
           uWS::OpCode opCode) {
#else
  h.onMessage(
      [&pid, &speed_limiter](
           uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
           uWS::OpCode opCode) {
#endif

// "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {
      auto s = hasData(string(data).substr(0, length));

      if (s != "") {
        auto j = json::parse(s);

        string event = j[0].get<string>();

        if (event == "telemetry") {
          // j[1] is the data JSON object
          double cte = std::stod(j[1]["cte"].get<string>());
          double speed = std::stod(j[1]["speed"].get<string>());
          //double angle = std::stod(j[1]["steering_angle"].get<string>());
          double steer_value;
          /**
           * OK: Calculate steering value here, remember the steering value is
           *   [-1, 1].
           * NOTE: Feel free to play around with the throttle and speed.
           *   Maybe use another PID controller to control the speed!
           */
#ifdef USE_TWIDDLE
          total_error += fabs(cte);
          count += 1;
          if (count == TWIDDLE_BATCH_SIZE) {
            // Get average error in batch
            double batch_error = total_error / count;

            // DEBUG
            std::cout << "========== End of batch ==========" << std::endl
                      << "Total error: " << total_error
                      << ", batch_error: " << batch_error
                      << ", param_idx: " << twiddle.param_idx
                      << std::endl;
            std::cout << "Best error: " << twiddle.best_err
                      << ", Kp: " << twiddle.p[0]
                      << ", Ki: " << twiddle.p[1]
                      << ", Kd: " << twiddle.p[2]
                      << std::endl;
          	std::cout << "CTE: " << cte
                      << ", p_error: " << pid.p_error
                      << ", i_error: " << pid.i_error
                      << ", d_error: " << pid.d_error
                      << ", total_error: " << pid.TotalError()
                      << std::endl;

            // Update params using twiddle
            twiddle.Update(batch_error);

            // Re-initialize PID with updated params
            pid.Init(twiddle.p[0], twiddle.p[1], twiddle.p[2]);

            // Reset batch error params
            total_error = 0;
            count = 0;
          }
#endif
          pid.UpdateError(cte);
          steer_value = -pid.TotalError();
          if (steer_value > 1) {
            steer_value = 1;
          } else if (steer_value < -1) {
            steer_value = -1;
          }

          speed_limiter = speed > speed_limiter ? MIN_SPEED : MAX_SPEED;
          double throttle = (
              1.0
              - pow(steer_value, 2)
              - pow(speed/speed_limiter, 2));

#ifndef USE_TWIDDLE
         std::cout << "CTE: " << cte
                   << ", Steering Value: " << steer_value
                   << ", p_error: " << pid.p_error
                   << ", i_error: " << pid.i_error
                   << ", d_error: " << pid.d_error
                   << ", total_error: " << pid.TotalError()
                   << std::endl;
#endif
          json msgJson;
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = throttle;
          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          //std::cout << msg << std::endl;
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket message if
  }); // end h.onMessage

  h.onConnection([](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  
  h.run();
}