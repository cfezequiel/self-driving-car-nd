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

#include "map.h"
#include "trajectory.h"
#include "vehicle.h"
#include "helpers.h"

#include <uWS/uWS.h>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"

#include <assert.h>
#include <fstream>
#include <iostream>
#include <string>

// for convenience
using nlohmann::json;
using std::string;
using std::vector;
using std::size_t;
using std::cout;
using std::endl;


int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  MapWaypoints map_waypoints;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  // double max_s    = 6945.554;

  std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    std::istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints.x.push_back(x);
    map_waypoints.y.push_back(y);
    map_waypoints.s.push_back(s);
    map_waypoints.dx.push_back(d_x);
    map_waypoints.dy.push_back(d_y);
  }

  // Start in lane 1
  int lane = 1;

  // Have a reference velocity to target
  double ref_vel = 0;

  h.onMessage([&ref_vel, &map_waypoints, &lane]
              (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
               uWS::OpCode opCode) {


    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);

        string event = j[0].get<string>();

        if (event == "telemetry") {
          // j[1] is the data JSON object

          // Set ego vehicle localization and sensor_fusion data
          EgoVehicle car(
              j[1]["x"],
              j[1]["y"],
              j[1]["s"],
              j[1]["d"],
              j[1]["yaw"],
              j[1]["speed"],
              // Previous path data given to the Planner
              j[1]["previous_path_x"],
              j[1]["previous_path_y"],
              // Previous path's end s and d values
              j[1]["end_path_s"],
              j[1]["end_path_d"],
              j[1]["sensor_fusion"],
              lane);

          if (car.VehicleInFrontTooClose()) {
            if (lane > 0 && car.LeftLaneClear()) {
              // Move left if clear
              lane -= 1;
            } else if (lane < 2 && car.RightLaneClear()) {
              // Move right if clear
              lane += 1;
            } else {
              // Slow down
              ref_vel -= 0.224;
            }
          } else {
            if (ref_vel < 49.5) {
              // Slowly accelerate if below target velocity
              // (useful for cold start)
              ref_vel += 0.224;
            }
          }

          // Get a trajectory based on target lane and velocity
          Trajectory trajectory = getTrajectory(
              lane, ref_vel, car, map_waypoints);

          json msgJson;
          msgJson["next_x"] = trajectory.x_vals;
          msgJson["next_y"] = trajectory.y_vals;

          auto msg = "42[\"control\","+ msgJson.dump()+"]";

          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket if
  }); // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
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