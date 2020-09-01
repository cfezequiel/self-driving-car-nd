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

"""Camera calibration"""

import glob
import logging
from typing import Dict, Tuple, Optional

import cv2
import numpy as np


def calibrate(file_pattern: str, nx: int, ny: int):
  """Returns camera calibration parameters from `cv2.calibrateCamera`.
  
  Args:
    file_pattern: Glob of calibration (i.e. chessboard pattern) images.
    nx: Number of calibration points per row.
    ny: Number of calibration points per column. 
    
  Returns:
    (ret, mtx, dist, rvecs, tvecs)
  """
  
  image_files = glob.glob(file_pattern)
  
  imgpoints = []
  objpoints = []
  objp = np.zeros((nx*ny, 3), np.float32)
  objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
  
  for fname in image_files:
    gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(
        gray, (nx, ny), None)
    
    if ret:
        imgpoints.append(corners)
        objpoints.append(objp)
    else:
        logging.warning(f'Could not find all corners for image {fname}.'
                          ' Not including any corners for this image')
        
  # Get height and width from last image
  height, width = gray.shape
  
  return cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
      
  