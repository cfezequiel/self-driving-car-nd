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

#ifndef VEHICLE_H
#define VEHICLE_H

#include <map>
#include <string>
#include <vector>

using std::map;
using std::size_t;
using std::string;
using std::vector;

// Default threshold distance (in meters) between the ego car
// and other vehicles whether the latter is too close to the former.
const unsigned int kMinGap = 30;


class Vehicle {
  public:
    double x;
    double y;
    double s;
    double d;

    Vehicle(const double& x, const double& y, const double& s, const double& d):
      x(x), y(y), s(s), d(d) {}
};


class SensedVehicle: public Vehicle {
public:
  double id;
  double vx;
  double vy;

  SensedVehicle(
      const double& x,
      const double& y,
      const double& s,
      const double& d,
      const double& id,
      const double& vx,
      const double& vy):
      Vehicle(x, y, s, d), id(id), vx(vx), vy(vy) {}
};


class EgoVehicle: public Vehicle {
  public:
    double yaw;
    double speed;
    vector<double> previous_path_x;
    vector<double> previous_path_y;
    double end_path_s;
    double end_path_d;
    vector<SensedVehicle> sensed_vehicles;
    int lane;

    EgoVehicle(
        const double& x,
        const double& y,
        const double& s,
        const double& d,
        const double& yaw,
        const double& speed,
        const vector<double>& previous_path_x,
        const vector<double>& previous_path_y,
        const double& end_path_s,
        const double& end_path_d,
        const vector< vector<double> >& sensor_fusion,
        const int& lane):
        Vehicle(x, y, s, d), yaw(yaw), speed(speed),
        previous_path_x(previous_path_x), previous_path_y(previous_path_y),
        end_path_s(end_path_s), end_path_d(end_path_d), lane(lane) {

      for (auto&& d: sensor_fusion) {
        SensedVehicle sensed_vehicle(
            d[1], d[2], d[5], d[6], d[0], d[3], d[4]);
        sensed_vehicles.push_back(sensed_vehicle);
      }
    }

    SensedVehicle* getVehicleNearby(
        const int& target_lane, const bool& ahead=true);
    SensedVehicle* getVehicleAhead(const int& lane);
    SensedVehicle* getVehicleBehind(const int& lane);
    bool VehicleTooCloseProjected(
        const int &target_lane,
        const bool& ahead=true,
        const unsigned int& threshold=kMinGap);
    bool VehicleInFrontTooClose(const unsigned int& threshold=kMinGap);
    bool LaneClear(
        const int& target_lane, const unsigned int& threshold=kMinGap);
    bool LeftLaneClear();
    bool RightLaneClear();
};


#endif // VEHICLE_H