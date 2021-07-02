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

#include "helpers.h"
#include "vehicle.h"

#include <iostream>
#include <limits>
#include "math.h"

using std::cout;
using std::endl;


// == EgoVehicle ==

SensedVehicle* EgoVehicle::getVehicleNearby(const int& target_lane, const bool& ahead) {
  int d_target = 2 + 4*target_lane;
  double min_dist = std::numeric_limits<double>::max();
  size_t min_i = -1;

  // check for nearby vehicle ahead (1) or in the rear (-1)
  int parity = ahead == true ? 1: -1;
  for (size_t i = 0; i < this->sensed_vehicles.size(); i++) {
    d = this->sensed_vehicles[i].d;
    // check cars in target lane
    if (d < d_target + 2 && d > d_target - 2) {
      SensedVehicle* sv = &this->sensed_vehicles[i];
      double dist = distance(sv->x, sv->y, this->x, this->y);
      if (parity*(sv->s - this->s) >= 0 && dist < min_dist) {
        min_dist = dist;
        min_i = i;
      }
    }
  }

  if (min_i >= 0) {
    return &this->sensed_vehicles[min_i];
  }

  return NULL;
}


SensedVehicle* EgoVehicle::getVehicleAhead(const int& lane) {
  return this->getVehicleNearby(lane);
}


SensedVehicle* EgoVehicle::getVehicleBehind(const int& lane) {
  return this->getVehicleNearby(lane, false);
}


bool EgoVehicle::VehicleTooCloseProjected(
    const int &target_lane, const bool& ahead, const unsigned int& threshold) {

  bool too_close = false;
  int parity = ahead == true ? 1: - 1;

  SensedVehicle* vehicle_nearby = this->getVehicleNearby(target_lane, ahead);

  if (vehicle_nearby != NULL) {
    // Get projected s of ego car
    size_t prev_size = this->previous_path_x.size();
    double projected_s;
    if (prev_size > 0) {
      projected_s = this->end_path_s;
    } else {
      projected_s = this->s;
    }

    // Get projected s of sensed vehicle
    double vx = vehicle_nearby->vx;
    double vy = vehicle_nearby->vy;
    double speed = sqrt(vx*vx + vy*vy);
    double projected_other_s = (
        vehicle_nearby->s + ((double) prev_size * 0.02 * speed));

    if (parity*(projected_other_s - projected_s) >= 0 &&
        fabs(projected_other_s - projected_s) <= threshold) {
      too_close = true;
    }
  }

  return too_close;
}


bool EgoVehicle::VehicleInFrontTooClose(const unsigned int& threshold) {
  return this->VehicleTooCloseProjected(this->lane, true, threshold);
}


bool EgoVehicle::LaneClear(
    const int& target_lane, const unsigned int& threshold) {

  // Check if the vehicles ahead and behind in target lane are not going to be
  // too close
  if (this->VehicleTooCloseProjected(target_lane, true, threshold) ||
      this->VehicleTooCloseProjected(target_lane, false, threshold)) {
    return false;
  }
  return true;
}


bool EgoVehicle::LeftLaneClear() {
  if (this->lane == 0) {
    return false;
  }

  return this->LaneClear(this->lane - 1);
}


bool EgoVehicle::RightLaneClear() {
  if (this->lane == 2) {
    return false;
  }

  return this->LaneClear(this->lane + 1);
}
