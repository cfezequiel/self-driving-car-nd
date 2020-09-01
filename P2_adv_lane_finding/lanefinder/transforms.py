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

"""Image transforms for lane finding."""

import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
from nptyping import NDArray
import numpy as np

from lanefinder import measure
from lanefinder import types
from lanefinder import utils


_LAB_YELLOW = utils.get_color((255, 255, 0), cv2.COLOR_RGB2LAB)
_LAB_WHITE = utils.get_color((255, 255, 255), cv2.COLOR_RGB2LAB)
_SEARCH_WINDOW_COLOR = (0, 255, 0)  # green
_LINE_THICKNESS = 2


def undistort(img: types.Image, cam: types.CameraParams) \
  -> types.Image:
  """Undistort image given camera calibration parameters."""
  
  return cv2.undistort(img, cam.mtx, cam.dist)


def color_threshold(
    img: types.RGBImage, 
    ref_color: Tuple[int, int, int],
    margin: Union[int, Tuple[int, int, int]] = (25, 25, 25), 
    colorspace: int = cv2.COLOR_RGB2LAB) -> types.BinaryImage:
  """Applies color thresholding in given colorspace.
  
  Args:
    img: RGB image.
    ref_color: Reference color in given colorspace.
    margin: Margin around `ref_color` for setting the threshold.
      i.e. threshold min = color - margin, threshold max = color + margin
    colorspace: OpenCV colorspace conversion flag.
      e.g. cv2.COLOR_RGB2LAB
  """
  
  img_ = cv2.cvtColor(img, colorspace)
  ref_color_ = np.array(ref_color, dtype=np.uint16)
  return cv2.inRange(img_, ref_color_ - margin, ref_color_ + margin)


def yellow_threshold(
    img: types.RGBImage, 
    margin: Union[int, Tuple[int, int, int]] = (25, 25, 25)) \
  -> types.BinaryImage:
  """Thresholds image against the color yellow in CIELAB colorspace."""
  
  return color_threshold(img, _LAB_YELLOW, margin=margin)


def channel_threshold(
    img: types.RGBImage,
    channel: int,
    thresh_min: int,
    thresh_max: int = 255,
    colorspace: int = None) -> types.BinaryImage:
  """Apply binary thresholding to an image channel."""
  
  if colorspace is not None:
    img = cv2.cvtColor(img, colorspace)
    
  channel_img = img[:, :, channel]
  return cv2.inRange(channel_img, thresh_min, thresh_max)


def white_threshold(
    img: types.RGBImage, 
    margin: Union[int, Tuple[int, int, int]] = (50, 50, 50)) \
  -> types.BinaryImage:
  """Thresholds image against the color white in CIELAB colorspace."""
  
  return color_threshold(img, _LAB_WHITE, margin=margin)


def gradient_threshold(
    img: types.RGBImage, 
    thresh_min: int = 30, 
    #thresh_min: int = 40, 
    thresh_max: int = 100) -> types.BinaryImage:
  """Returns gradient threshold binary image.
  
  This applies Sobel edge detection along the X direction of the image.
  
  Args:
    img: RGB image.
    thresh_min: Minimum pixel value where binary output pixel is 1.
    thresh_max: Maximum pixel value where binary output pixel is 1.
  """
  
  # Take the derivative in x using Sobel
  sobelx = cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F, 1, 0)
  abs_sobelx = np.absolute(sobelx)
  scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
  
  # Threshold x gradient
  sxbinary = np.zeros_like(scaled_sobel)
  sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 255
  
  return sxbinary


def apply_thresholds(
    img: types.RGBImage, 
    threshold_fns: List[Callable[[types.RGBImage, Optional[int]], types.BinaryImage]]) \
  -> types.BinaryImage:
  
  assert len(img.shape) == 3
  
  out_img = np.zeros(img.shape[:-1], dtype=np.uint8)
  for fn in threshold_fns:
    mask = fn(img)
    out_img = cv2.bitwise_or(out_img, mask)
    
  return out_img 


def perspective_transform(
    img: types.Image,
    src_points: NDArray[(4, 4), np.uint32], 
    dst_points: NDArray[(4, 4), np.uint32]) -> types.Image:
  """Applies perspective transform to given image.
  
  Args:
    img: RGB, grayscale or binary, image.
    src_points: (x, y) control points to tranform.
    dst_points: (x, y) destination for control points.
    
  Returns:
    Perspective-transformed image.
  """
  
  transform_mtx = cv2.getPerspectiveTransform(src_points, dst_points)
  y_max, x_max = img.shape[0], img.shape[1]
  return cv2.warpPerspective(img, transform_mtx, (x_max, y_max))


def overlay_polygon(
    img: types.Image, 
    vertices: types.Coordinates2D, 
    color: Optional[types.ColorRGB] = (0, 255, 255),
    thickness: Optional[int] = 5) -> types.RGBImage:
  """Draw polygon over image."""
  
  if len(img.shape) == 2:  # Assume gray
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
  
  pts = vertices.reshape((-1, 1, 2)).astype(np.int32)
  overlay_img = np.copy(img)
  cv2.polylines(
      overlay_img, [pts], isClosed=True, color=color, thickness=thickness)
  return overlay_img


def overlay_detected_lanes(
    img: types.BinaryImage, 
    lp: types.LanePixels, 
    lsw: Optional[types.LaneSearchWindows] = None,
    flp: Optional[types.FittedLaneParams] = None,
    margin: int = 50) \
  -> types.RGBImage:
  """Plots detected lane pixels with search boundary."""
  
  plot_img = np.dstack((img,)*3)

  # Colors in the left and right lane regions
  plot_img[lp.lefty, lp.leftx] = [255, 0, 0]
  plot_img[lp.righty, lp.rightx] = [0, 0, 255]

  y_max = img.shape[0]
  if flp:
    if margin is None:
      raise ValueError('Specify margin when `flp` is specified.')
      
    left_fitx, right_fitx, ploty = measure.get_poly_points(flp, y_max)
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    window_img = np.zeros_like(plot_img)
    cv2.fillPoly(window_img, np.int_([left_line_pts]), _SEARCH_WINDOW_COLOR)
    cv2.fillPoly(window_img, np.int_([right_line_pts]), _SEARCH_WINDOW_COLOR)
    plot_img = cv2.addWeighted(plot_img, 1, window_img, 0.3, 0)
    
  else:
    # Fit lane pixels to lane lines
    flp = measure.fit_poly(lp)
    left_fitx, right_fitx, ploty = measure.get_poly_points(flp, y_max)
    
    if lsw:
      # Draw the windows on the visualization image
      for i in range(len(lsw.left)):
        cv2.rectangle(
            plot_img, 
            (lsw.left[i].x_low, lsw.left[i].y_low), 
            (lsw.left[i].x_high, lsw.left[i].y_high), 
            _SEARCH_WINDOW_COLOR, _LINE_THICKNESS) 
        cv2.rectangle(
            plot_img, 
            (lsw.right[i].x_low, lsw.right[i].y_low), 
            (lsw.right[i].x_high, lsw.right[i].y_high), 
            _SEARCH_WINDOW_COLOR, _LINE_THICKNESS) 

  # Draw the fitted lane lines
  left_pts = np.array(list(zip(left_fitx, ploty))).reshape(-1, 1, 2).astype(np.uint64)
  right_pts = np.array(list(zip(right_fitx, ploty))).reshape(-1, 1, 2).astype(np.uint64)
  return cv2.polylines(
      plot_img, [left_pts, right_pts], isClosed=False, color=[255, 255, 0], thickness=2)
  

def overlay_lane_polygon(
    img: types.RGBImage,
    flp: types.FittedLaneParams,
    inv_transform_mtx: NDArray[(3, 3), np.float64]) \
  -> types.RGBImage:
  """Overlays detected lane on an image."""
  
  # Get lane pixels using a prior polynomial fit
  y_max = img.shape[0]
  x_max = img.shape[1]
  left_fitx, right_fitx, ploty = measure.get_poly_points(flp, y_max)

  # Create an image to draw the lines on
  blank_img = np.zeros_like(img[:, :, 0]).astype(np.uint8)
  color_warp_img = np.dstack((blank_img,)*3)

  # Recast the x and y points into usable format for cv2.fillPoly()
  pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
  pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
  pts = np.hstack((pts_left, pts_right))

  # Draw the lane onto the warped blank image
  cv2.fillPoly(color_warp_img, np.int_([pts]), (0, 255, 0))

  # Warp the blank back to original image space using inverse perspective matrix
  unwarped_img = cv2.warpPerspective(color_warp_img, inv_transform_mtx, (x_max, y_max)) 
  
  # Combine the result with the original image
  return cv2.addWeighted(img, 1, unwarped_img, 0.3, 0)


def detect_lane_boundary(
    img, 
    warped_img, 
    inv_transform_mtx,
    ym_per_pix: float = 1.,
    xm_per_pix: float = 1.,
    flp: Optional[types.FittedLaneParams] = None,
    min_pixels: int = 5000,
    new_flp_weight: float = 0.20,
  ) -> Tuple[types.RGBImage, types.FittedLaneParams]:
  """Detect lane boundary and overlay on image.
  
  This displays vehicle offset and radius of curvature of the detected lane
  as well.
  """
  
  # Find lane pixels and optionally, lane boundaries
  if flp is not None:
    lp = measure.find_lane_pixels_from_prior(warped_img, flp)
  else:
    lp, _ = measure.find_lane_pixels(warped_img)
    
  # Fit new lane lines based on detected pixels
  new_flp = measure.fit_poly(lp)
  if flp:
    # Get moving average of fitted lane parameters
    if len(lp.leftx) > min_pixels:
      flp.left = new_flp.left
    else:
      flp.left = flp.left*(1 - new_flp_weight) + new_flp.left*new_flp_weight
    if len(lp.rightx) > min_pixels:
      flp.right = new_flp.right
      flp.right = flp.right*(1 - new_flp_weight) + new_flp.right*new_flp_weight
  else:
    flp = new_flp

  # Get radius of curvature
  y_max = warped_img.shape[0]
  left_fitx, right_fitx, ploty = measure.get_poly_points(flp, y_max)
  left_roc, right_roc = measure.lane_curvature(
      left_fitx, right_fitx, ploty, ym_per_pix, xm_per_pix)
  roc = (left_roc + right_roc) / 2

  # Get vehicle offset from lane center
  vehicle_offset = measure.vehicle_offset(
      warped_img, left_fitx[-1], right_fitx[-1], xm_per_pix)

  # Overlay lane boundary
  overlay_img = overlay_lane_polygon(img, flp, inv_transform_mtx)

  # Add ROC and vehicle offset information
  vehicle_rel_pos = 'left' if vehicle_offset < 0 else 'right'
  roc_text = f'Radius of curvature = {roc:.2f}m'
  vehicle_offset_text = (
      f'Vehicle is {abs(vehicle_offset):.2f}m'
      f' {vehicle_rel_pos} of center'
  )
  put_text = functools.partial(
    cv2.putText, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
    color=[255, 255, 255], thickness=2)
  out_img = put_text(overlay_img, roc_text, (400, 50))
  out_img = put_text(out_img, vehicle_offset_text, (400, 100))
  
  return out_img, flp
  

def region_of_interest(img: types.Image, vertices: types.Coordinates2D):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    vertices = vertices.reshape((-1, 1, 2)).astype(np.int32)
    
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, [vertices], ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    roi_img = cv2.bitwise_and(img, mask)
    return roi_img
  

def equalize_hist(img: types.Image):
  """Applies histogram equalization to each channel of an RGB image."""
  
  if len(img.shape) > 2:
    channels = np.split(img, img.shape[-1], axis=-1)
  else:
    channels = [img]
  
  out_channels = []
  for channel in channels:
    out_channels.append(cv2.equalizeHist(channel))
  return np.dstack(out_channels)
