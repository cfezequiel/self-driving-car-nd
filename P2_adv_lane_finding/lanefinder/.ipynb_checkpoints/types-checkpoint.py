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

"""Defined data types."""

import collections
import dataclasses
from typing import Any, List, Tuple

import numpy as np
from nptyping import NDArray


CameraMatrix = NDArray[(3, 3), np.float64]
DistortionCoeff = NDArray[(1, 5), np.float64]
Image = NDArray[(Any, Any, Any), np.uint8]
RGBImage = NDArray[(Any, Any, 3), np.uint8]
BinaryImage = NDArray[(Any, Any), np.uint8]
Histogram = NDArray[(Any), np.int32]
Coordinates1D = NDArray[(Any), np.int64] 
Coordinates2D = NDArray[(Any, 2), np.float32]
ColorRGB = Tuple[int, int, int]
QuadraticParams = NDArray[(3), np.float64]


@dataclasses.dataclass
class CameraParams:
  
  mtx: CameraMatrix
  dist: DistortionCoeff
    

@dataclasses.dataclass
class WarpParams:
  
  src_points: Coordinates2D
  dst_points: Coordinates2D
    
    
@dataclasses.dataclass
class MetersPerPixel:
  
  x: float
  y: float


@dataclasses.dataclass
class LanePixels:
  """Coordinates for left and right lane pixels."""
  
  leftx: Coordinates1D
  lefty: Coordinates1D
  rightx: Coordinates1D
  righty: Coordinates1D
    

@dataclasses.dataclass
class RectangleCoords:
  """Coordinates to form a rectangle on an image."""
  
  y_low: int
  y_high: int
  x_low: int
  x_high: int
    
    
@dataclasses.dataclass
class LaneSearchWindows:
  
  left: List[RectangleCoords] = dataclasses.field(default_factory=list)
  right: List[RectangleCoords] = dataclasses.field(default_factory=list)
 

@dataclasses.dataclass
class FittedLaneParams:
   
    left: QuadraticParams
    right: QuadraticParams