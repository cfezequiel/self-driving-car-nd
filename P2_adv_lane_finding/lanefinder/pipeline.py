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

"""Image processing pipeline for finding lanes."""

import functools
from typing import Callable, Dict, Optional

import cv2
import numpy as np

from lanefinder import measure
from lanefinder import transforms
from lanefinder import types


MOVING_AVG_WINDOW = 100

WARP_PARAMS = types.WarpParams(
  src_points=np.float32([
    #[223, 680],
    [223, 720],
    [548, 460],
    [764, 460],
    #[1110, 680]]),
    #[1110, 720]]),
    [1130, 720]]),
  dst_points=np.float32([
    [400, 720],
    [400, 0],
    [920, 0],
    [920, 720]]),
)

METERS_PER_PIXEL = types.MetersPerPixel(
    x=3.7/700,
    y=30/720,
)

THRESHOLD_FN_MAP = {
    'gradient': transforms.gradient_threshold,
    'white': transforms.white_threshold,
    'yellow': functools.partial(
        transforms.yellow_threshold,
        margin=(200, 200, 75)),
}


class Pipeline:
  """Lane detection pipeline."""
  
  _moving_avg_prop = 1. / MOVING_AVG_WINDOW
  
  def __init__(
      self, 
      cam: types.CameraParams,
      threshold_fn_map: Optional[Dict[str, Callable]] = None,
      warp_params: Optional[types.WarpParams] = None,
      mpp: Optional[types.MetersPerPixel] = None):
    """Initialize."""
    
    self.cam = cam
    self.threshold_fn_map = (
        threshold_fn_map if threshold_fn_map is not None else THRESHOLD_FN_MAP)
    self.warp_params = warp_params or WARP_PARAMS
    self.mpp = mpp or METERS_PER_PIXEL
    
    # types.FittedLaneParams
    self.flp = None
    
  def _update_lane_fit(self, flp):
    """Updates the lefts and right lane boundary params."""
    
    if self.flp:
      self.flp.left = (
          self.flp.left*(1 - self._moving_avg_prop) +  
          flp.left*self._moving_avg_prop)
      
      self.flp.right = (
          self.flp.right*(1 - self._moving_avg_prop) + 
          flp.right*self._moving_avg_prop)
    else:
      self.flp = flp
      
  def reset(self):
    """Resets pipeline state params."""
    
    self.flp = None
    
  def process(self, img, analyze=False):
    """Runs the pipeline on an image."""
    
    # Undistort image
    undist_img = transforms.undistort(img, self.cam)
    
    # Apply color/gradient thresholding
    binary_img = transforms.apply_thresholds(
        undist_img, self.threshold_fn_map.values())
    
    # Apply ROI filter
    src_points = self.warp_params.src_points
    unwarped_roi_img = transforms.region_of_interest(binary_img, src_points)
    
    # Perspective transform image (i.e bird's eye view)
    dst_points = self.warp_params.dst_points
    warped_img = transforms.perspective_transform(
        unwarped_roi_img, self.warp_params.src_points, self.warp_params.dst_points)
    
    # Apply ROI localization
    roi_img = transforms.region_of_interest(warped_img, dst_points)
    
    # Used only when analyze=True
    # If fitted lane parameters have not been set (i.e. first image in pipeline),
    # use lane search windows detectoin and overlay windows on the warped img
    use_search_windows = True if not self.flp else False
    
    # Detect lane boundary and overlay on undistored image
    inv_transform_mtx = cv2.getPerspectiveTransform(
        self.warp_params.dst_points, self.warp_params.src_points)
    out_img, flp = transforms.detect_lane_boundary(
        undist_img, roi_img, inv_transform_mtx, 
        self.mpp.y, self.mpp.x, self.flp)
    self._update_lane_fit(flp)
    
    if analyze:
      out_imgs = {
          'input': img,
          'undistorted': undist_img,
      }
      
      for label, fn in self.threshold_fn_map.items():
        out_imgs[f'threshold-{label}'] = fn(undist_img)
      
      vertices_img = np.dstack((binary_img,)*3)
      vertices_img = transforms.overlay_polygon(vertices_img, src_points)
      vertices_img = transforms.overlay_polygon(
          vertices_img, dst_points, color=transforms._SEARCH_WINDOW_COLOR)
      
      lp, lsw = measure.find_lane_pixels(roi_img)
      if use_search_windows:
        lane_img = transforms.overlay_detected_lanes(roi_img, lp, lsw)
      else:
        lane_img = transforms.overlay_detected_lanes(roi_img, lp, flp=self.flp)
        
      binary_roi_overlay_img = transforms.overlay_polygon(binary_img, src_points)
      warped_roi_overlay_img = transforms.overlay_polygon(warped_img, dst_points)
      
      out_imgs.update({
          'undistorted': undist_img,
          'binary': binary_roi_overlay_img,
          'unwarped ROI': unwarped_roi_img,
          'warped': warped_roi_overlay_img,
          'warped ROI': roi_img,
          'detection': lane_img,
          'output': out_img,
      })
      return out_imgs
    
    return out_img
  