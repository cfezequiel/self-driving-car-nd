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

 Copyright (c) 2016-2018 Udacity, Inc.

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

/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * OK: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * OK: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  this->num_particles = 100;
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  for (std::size_t i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1;
    this->weights.push_back(1);
    this->particles.push_back(p);
  }

  this->is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * OK: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);

  double c, dx, dy, dtheta;
  for (std::size_t i = 0; i < this->particles.size(); i++) {
    Particle p = this->particles[i];
    if (fabs(yaw_rate) > 0.0001) {
      c = velocity/yaw_rate;
      dtheta = yaw_rate*delta_t;
      dx = c*(sin(p.theta + dtheta) - sin(p.theta));
      dy = c*(cos(p.theta) - cos(p.theta + dtheta));
    }
    else {
      c = velocity*delta_t;
      dtheta = 0;
      dx = c*cos(p.theta);
      dy = c*sin(p.theta);
    }

    // Add random Gaussian noise to the predictions
    this->particles[i].x += dx + dist_x(gen);
    this->particles[i].y += dy + dist_y(gen);
    this->particles[i].theta += dtheta + dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
  /**
   * OK: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */

  // Implement nearest-neighbors algorithm O(m*n)
  for (std::size_t i = 0; i < observations.size(); i++) {

    LandmarkObs ob = observations[i];
    double nearest = std::numeric_limits<double>::max();
    int nearest_id = -1;
    double nearest_x, nearest_y;
    
    for (std::size_t j = 0; j < predicted.size(); j++) {
      LandmarkObs pr = predicted[j];
      double distance = dist(pr.x, pr.y, ob.x, ob.y);
      if (distance < nearest) {
        nearest = distance;
        nearest_id = pr.id;
        nearest_x = pr.x;
        nearest_y = pr.y;
      }
    }
    observations[i].id = nearest_id;
    
    // As an optimization, compute the coordinate differences between the observation 
    // and predicted landmark here, so that the predicted coordinates need not be
    // searched for later on in `updateWeights`.
    if (nearest_id >= 0) {
      observations[i].x -= nearest_x;
      observations[i].y -= nearest_y;
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * OK: Update the weights of each particle using a multi-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for (std::size_t i = 0; i < this->num_particles; i++) {
    Particle p = this->particles[i];

    // "Predict" landmark measurements for each particle
    vector<LandmarkObs> predicted;
    for (std::size_t j = 0; j < map_landmarks.landmark_list.size(); j++) {
      int lm_id = map_landmarks.landmark_list[j].id_i;
      float lm_x = map_landmarks.landmark_list[j].x_f;
      float lm_y = map_landmarks.landmark_list[j].y_f;
      if (fabs(lm_x - p.x) <= sensor_range && fabs(lm_y - p.y) <= sensor_range) {
        predicted.push_back(LandmarkObs {lm_id, lm_x, lm_y});
      }
    }

    // Convert observations into map coordinates
    vector<LandmarkObs> map_obs;
    for (std::size_t j = 0; j < observations.size(); j++) {
      LandmarkObs ob = observations[j];
      double x_m = ob.x*cos(p.theta) - ob.y*sin(p.theta) + p.x;
      double y_m = ob.x*sin(p.theta) + ob.y*cos(p.theta) + p.y;
      map_obs.push_back(LandmarkObs {ob.id, x_m, y_m});
    }

    // Associate each observation with the closest landmark
    dataAssociation(predicted, map_obs);

    // Calculate multivariate gaussian PDF between landmark coordinates and
    //   observation coordinates
    double weight = 1;
    for (std::size_t j = 0; j < map_obs.size(); j++) {
      LandmarkObs ob = map_obs[j];
      // Note: `ob.x` and `ob.y` should already be offsets against the 
      // matching predicted landmark's coordinates, e.g.
      // ob.x = obs.x - predicted.x
      // from the `dataAssociation` function.
      double prob = pdf2d(
          ob.x, ob.y, 0, 0, std_landmark[0], std_landmark[1]);
      weight *= prob;
    }
	
    // Update particle weight
    p.weight = weight;
    weights[i] = weight;
  }
}

void ParticleFilter::resample() {
  /**
   * OK: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::default_random_engine gen;
  std::discrete_distribution<std::size_t> dist(this->weights.begin(), this->weights.end());

  std::vector<Particle> new_particles;
  for (std::size_t i = 0; i < this->num_particles; i++) {
    Particle p = this->particles[dist(gen)];
    new_particles.push_back(p);
  }
  this->particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}