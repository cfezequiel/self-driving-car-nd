# Lint as: Python3

# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file incorporates work covered by the following copyright and  
# permission notice:  
#
# MIT License
#
# Copyright (c) 2016-2019 Udacity, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Misc. utility functions."""

from typing import Tuple

import cv2
import numpy as np
import nptyping


def get_color(rgb: Tuple[int, int, int], colorspace_convert: int) \
  -> Tuple[int, int, int]:
  rgb_img = np.array(rgb).reshape(1, 1, 3).astype(np.uint8)
  return tuple(cv2.cvtColor(rgb_img, colorspace_convert)[0][0])


def generate_data(ym_per_pix, xm_per_pix):
    """Generates fake data to use for calculating lane curvature."""
    
    # Set random seed number so results are consistent for grader
    # Comment this out if you'd like to see results on different random data!
    np.random.seed(0)
    
    # Generate some fake data to represent lane-line pixels
    ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
    quadratic_coeff = 3e-4 # arbitrary quadratic coefficient
    
    # For each y position generate random x position within +/-50 pix
    # of the line base position in each case (x=200 for left, and x=900 for right)
    leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                    for y in ploty])
    rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                    for y in ploty])

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    # Fit a second order polynomial to pixel positions in each fake lane line
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    
    return ploty, left_fit_cr, right_fit_cr
  

def corners_unwarp(img, nx, ny, mtx, dist):
  """Undistorts a chessboard image."""
  
  undist_img = cv2.undistort(img, mtx, dist)
  gray_img = cv2.cvtColor(undist_img, cv2.COLOR_RGB2GRAY)
  ret, corners = cv2.findChessboardCorners(gray_img, (nx, ny))
  cv2.drawChessboardCorners(undist_img, (nx, ny), corners, ret)
  src_indexes = [0, nx - 1, nx*(ny - 1), nx*ny - 1]
  src_points = np.float32([corners[i][0] for i in src_indexes])
  ymax, xmax = img.shape[0], img.shape[1]
  delta = 0.05
  dy = ymax*delta
  dx = xmax*delta
  dst_points = np.float32([
      [dx, dy], 
      [xmax - dx, dy],
      [dx, ymax - dy],
      [xmax - dx, ymax - dy],
  ])
  M = cv2.getPerspectiveTransform(src_points, dst_points)
  warped_img = cv2.warpPerspective(undist_img, M, (xmax, ymax))
  return warped_img