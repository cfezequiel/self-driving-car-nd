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

#include "trajectory.h"
#include "helpers.h"
#include "map.h"
#include "spline.h"

using std::size_t;

const unsigned int kTotalPoints= 50;

Trajectory getTrajectory(
    const int& target_lane,
    const float& target_velocity,
    const EgoVehicle& car,
    const MapWaypoints& map_waypoints) {

  assert(target_lane >= 0);
  assert(target_lane <= 2);
  assert(target_velocity >= 0);

  // Create list of widely spaced (x,y) waypoint, evenly spaced at 30m
  // Later, interpolate these waypoints with a spline and fill it in with more
  // points that control speed
  vector<double> ptsx;
  vector<double> ptsy;

  // Reference x,y,yaw states
  // Represents either the starting point of the car or end point of the
  //   previous path
  double ref_x = car.x;
  double ref_y = car.y;
  double ref_yaw = deg2rad(car.yaw);

  // If the previous state is almost empty, use the car as starting reference
  size_t prev_size = car.previous_path_x.size();
  if (prev_size < 2) {
    // Use two points that make the path tangent to the car
    double prev_car_x = car.x - cos(car.yaw);
    double prev_car_y = car.y - sin(car.yaw);

    ptsx.push_back(prev_car_x);
    ptsy.push_back(prev_car_y);

    ptsx.push_back(car.x);
    ptsy.push_back(car.y);

  } else {
    // Use the previous path's end point as a starting reference
    // Use two points that make the path tangent to the previous path's end point

    double ref_x_prev = car.previous_path_x[prev_size - 2];
    double ref_y_prev = car.previous_path_y[prev_size - 2];

    ptsx.push_back(ref_x_prev);
    ptsy.push_back(ref_y_prev);

    ref_x = car.previous_path_x[prev_size - 1];
    ref_y = car.previous_path_y[prev_size - 1];

    ptsx.push_back(ref_x);
    ptsy.push_back(ref_y);

    ref_yaw = atan2(ref_y - ref_y_prev, ref_x - ref_x_prev);
  }

  // In Frenet add evenly 30m spaced points ahead of the starting reference
  int lane_d = 2 + 4*target_lane;
  double projected_s = prev_size > 0 ? car.end_path_s: car.s;
  for (size_t i = 1; i <= 3; i++) {
      vector<double> xy = getXY(
          projected_s + i*30, lane_d, map_waypoints.s, map_waypoints.x,
          map_waypoints.y);
      ptsx.push_back(xy[0]);
      ptsy.push_back(xy[1]);
  }

  // Shift from map coordinates to vehicle coordinates
  for (int i = 0; i < ptsx.size(); i++) {
    // shift car reference angle to 0 degrees
    double shift_x = ptsx[i] - ref_x;
    double shift_y = ptsy[i] - ref_y;

    // Rotational shift
    ptsx[i] = shift_x*cos(-ref_yaw) - shift_y*sin(-ref_yaw);
    ptsy[i] = shift_x*sin(-ref_yaw) + shift_y*cos(-ref_yaw);
  }

  // Create a spline
  tk::spline s;

  // Set (x,y) points to the spline
  s.set_points(ptsx, ptsy);

  vector<double> next_x_vals;
  vector<double> next_y_vals;

  // Add any remaining points from previous path
  for (int i = 0; i < car.previous_path_x.size(); i++) {
    next_x_vals.push_back(car.previous_path_x[i]);
    next_y_vals.push_back(car.previous_path_y[i]);
  }

  // Calculate how to break up spline points so that the
  //   car travels at desired reference velocity
  double target_x = 30.0;  // meters
  double target_y = s(target_x);
  double target_dist = sqrt(target_x*target_x + target_y*target_y);
  double x_add_on = 0;

  // Fill up the rest of the path
  double N = target_dist/(0.02*target_velocity/2.24);
  double target_inc = target_x/N;
  for (size_t i = 1; i <= kTotalPoints - car.previous_path_x.size(); i++) {
    double x_point = x_add_on + target_inc;
    double y_point = s(x_point);
    x_add_on = x_point;

    // Rotate back to normal after rotating earlier
    double x_ref = x_point;
    double y_ref = y_point;
    x_point = x_ref*cos(ref_yaw) - y_ref*sin(ref_yaw);
    y_point = x_ref*sin(ref_yaw) + y_ref*cos(ref_yaw);
    x_point += ref_x;
    y_point += ref_y;

    //std::cout << x_point << y_point << std::endl;
    next_x_vals.push_back(x_point);
    next_y_vals.push_back(y_point);
  }

  Trajectory trajectory;
  trajectory.x_vals = next_x_vals;
  trajectory.y_vals = next_y_vals;

  return trajectory;
}


