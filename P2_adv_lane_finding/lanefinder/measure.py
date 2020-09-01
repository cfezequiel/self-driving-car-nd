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

"""Lane extraction and measurement."""

from typing import Any, List, Tuple, Union

from nptyping import NDArray
import numpy as np

from lanefinder import types


def lane_histogram(img: types.BinaryImage) -> types.Histogram:
  """Compute a histogram of white pixels along the x-axis."""
  
  y_end = img.shape[0]
  y_start = y_end // 2
  bottom_half = img[y_start:y_end, :]
  histogram = np.sum(bottom_half, axis=0)
  return histogram


def lane_base_positions(
    img: types.BinaryImage,
    return_hist: bool = False) \
  -> Union[Tuple[int, int], Tuple[int, int, types.Histogram]]:
  """Returns x coordinates of the left and right lane lines.
  
  This infers the bottom-most x coordinates of the left and right lane lines 
  by first computing a histogram of the binary warped image, reducing the y-axis,
  and then choosing the x coordinates of the peaks (most number of white pixels).
  
  Args:
    img: Warped binary image.
    return_hist: If True, also return the computed histogram.
  """
  hist = lane_histogram(img)
  x_mid = img.shape[1] // 2
  y_max = img.shape[0]
  left_x = np.argmax(hist[:x_mid])
  right_x = x_mid + np.argmax(hist[x_mid:])
  
  if return_hist:
    return left_x, right_x, hist
  
  return left_x, right_x
           
           
def find_lane_pixels(
    img: types.BinaryImage,
    nwindows: int = 10, 
    margin: int = 50, 
    minpix: int = 100) \
  -> Tuple[types.LanePixels, types.LaneSearchWindows]:
  """Returns pixels belong to the left and right lane lines.
  
  This uses the sliding window approach.
  
  Args:
    img: Binary warped image.
    nwindows: Number of windows to use for searching.
    margin: one-half of the width of each window.
      (2*margin is the window width).
    minpix: Minimum number of white pixels needed in order to recalculate
      next window's center position (mean of the pixels).
  """
  
  y_max = img.shape[0]
  window_height = np.int(y_max // nwindows)
  nonzero = img.nonzero()
  nonzeroy, nonzerox = nonzero
  
  x_max = img.shape[1]
  x_mid = x_max // 2
  hist = lane_histogram(img)
  leftx_base = np.argmax(hist[:x_mid])
  rightx_base = x_mid + np.argmax(hist[x_mid:])
  
  leftx_current = leftx_base
  rightx_current = rightx_base
  
  left_lane_inds = []
  right_lane_inds = []
  
  search_windows = types.LaneSearchWindows()
  
  for window in range(nwindows):
    # Find window boundary points along y axis
    win_y_low = y_max - (window + 1)*window_height
    win_y_high = y_max - window*window_height
    
    # Find window boundary points along x axis
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    
    search_windows.left.append(types.RectangleCoords(
        win_y_low, win_y_high, win_xleft_low, win_xleft_high))
    search_windows.right.append(types.RectangleCoords(
        win_y_low, win_y_high, win_xright_low, win_xright_high))
    
    # Identify the nonzero pixels in x and y within the window
    good_left_inds = (
      (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &  
      (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = (
      (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
      (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
    
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
    
    # If you found > minpix pixels, recenter next window
    # TODO: make this robust to outliers
    if len(good_left_inds) > minpix:
      leftx_current_ = np.mean(nonzerox[good_left_inds]).astype(np.int)
      #leftx_current = leftx_current * 0.99 + leftx_current_ * 0.01
      leftx_current = leftx_current_
      
    if len(good_right_inds) > minpix:
      rightx_current_ = np.mean(nonzerox[good_right_inds]).astype(np.int)
      rightx_current = rightx_current_
    
  # Concatenate the arrays of indices (previously was a list of lists of pixels)
  try:
      left_lane_inds = np.concatenate(left_lane_inds)
      right_lane_inds = np.concatenate(right_lane_inds)
  except ValueError:
      # Avoids an error if the above is not implemented fully
      pass

  # Extract left and right line pixel positions
  leftx = nonzerox[left_lane_inds]
  lefty = nonzeroy[left_lane_inds] 
  rightx = nonzerox[right_lane_inds]
  righty = nonzeroy[right_lane_inds]
  
  return (types.LanePixels(leftx, lefty, rightx, righty),
          search_windows)


def find_lane_pixels_from_prior(
    img: types.BinaryImage,
    flp: types.FittedLaneParams,
    margin: int = 50) \
  -> types.LanePixels:
  """Returns lane pixels based on prior lane polynomial fit."""
  
  # Grab activated pixels
  nonzero = img.nonzero()
  nonzeroy = np.array(nonzero[0])
  nonzerox = np.array(nonzero[1])

  # Set the area of search based on activated x-values
  left_polyx = flp.left[0]*nonzeroy**2 + flp.left[1]*nonzeroy + flp.left[2]
  right_polyx = flp.right[0]*nonzeroy**2 + flp.right[1]*nonzeroy + flp.right[2]
  left_lane_inds = (
      (nonzerox > left_polyx - margin) & (nonzerox < left_polyx + margin)).nonzero()
  right_lane_inds = (
      (nonzerox > right_polyx - margin) & (nonzerox < right_polyx + margin)).nonzero()

  # Again, extract left and right line pixel positions
  leftx = nonzerox[left_lane_inds]
  lefty = nonzeroy[left_lane_inds] 
  rightx = nonzerox[right_lane_inds]
  righty = nonzeroy[right_lane_inds]
  
  return types.LanePixels(leftx, lefty, rightx, righty)


def fit_poly(lp: types.LanePixels) -> types.FittedLaneParams:
  """Fits a second order polynomial to left and right lane pixels."""
  
  if len(lp.lefty):
    left_fit = np.polyfit(lp.lefty, lp.leftx, 2)
  else:
    left_fit = 0.
  if len(lp.righty):
    right_fit = np.polyfit(lp.righty, lp.rightx, 2)
  else:
    right_fit = 0.
  return types.FittedLaneParams(left_fit, right_fit)


def get_poly_points(flp: types.FittedLaneParams, y_max: int) \
  -> Tuple[
      types.Coordinates1D,
      types.Coordinates1D,
      types.Coordinates1D]:
  """Returns x-y coordinates of pixels along the fitted lane lines."""
  ploty = np.linspace(0, y_max - 1, y_max )
  left_fit = flp.left
  right_fit = flp.right
  try:
      left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
      right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
  except TypeError:
      # Avoids an error if `left` and `right_fit` are still none or incorrect
      print('The function failed to fit a line!')
      left_fitx = 1*ploty**2 + 1*ploty
      right_fitx = 1*ploty**2 + 1*ploty
      
  return left_fitx, right_fitx, ploty

           
def radius_of_curvature(
    polyfit: NDArray[(3), np.float64], 
    ym_per_pix: float, 
    y_ref: int):
  """Returns the radius of curvature.
  
  Args:
    polyfit: 2nd-order polynomial parameters.
    ym_per_pix: Meters per pixel along the y-axis of the image.
    y_ref: Radius  of curvature reference point along y-axis 
  """
  # Assume 2nd order polynomial
  n = len(polyfit)
  if n != 3:
    raise ValueError(f'Expected 2nd order polynomial (3 params). Got {n} params.')
    
  return (((1 + (2*polyfit[0]*y_ref*ym_per_pix + polyfit[1])**2)**1.5) /
          np.absolute(2*polyfit[0]))


def lane_curvature(left_fitx, right_fitx, ploty, ym_per_pix, xm_per_pix):
    """Calculates the curvature of polynomial functions in meters."""
    
    # Fit new polynomials to x,y coordinates in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_ref = ploty[-1]
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = radius_of_curvature(left_fit_cr, ym_per_pix, y_ref)
    right_curverad = radius_of_curvature(right_fit_cr, ym_per_pix, y_ref)
    
    return left_curverad, right_curverad
  
  
def vehicle_offset(img, left_x, right_x, xm_per_pix):
  """Measures offset of the vehicle from the center of the lane."""
  
  lane_center = (left_x + right_x) / 2
  x_max = img.shape[1]
  vehicle_center = x_max / 2
  return xm_per_pix * (vehicle_center - lane_center)