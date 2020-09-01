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

"""Utility functions for image visualization in Jupyter."""

import math
from typing import Dict, Tuple, Optional

import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage import metrics as skmetrics

from lanefinder import measure
from lanefinder import types


def _gray(img: np.ndarray):
  """Converts image to grascale, if it is in another colorspace."""
  
  if len(img.shape) == 2:
    return img
  
  return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def imshow(img: np.ndarray, ax: plt.axis = None):
  """Displays image; autocorrects color map for grayscale."""
  
  if ax:
    imshow_fn = ax.imshow
  else:
    imshow_fn = plt.imshow
  
  n = len(img.shape)
  if n == 2 or img.shape[-1] == 1:
    imshow_fn(img, cmap='gray')
  else:
    imshow_fn(img)
    

def compare_images(
    in_img: np.ndarray, 
    plot_img: np.ndarray, 
    in_title: str = 'Input', 
    out_title: str = 'Output', 
    diff: bool = False, 
    figsize: Tuple[int, int] = (20, 10), 
    axis_off: bool = True, 
    return_fig: bool = False):
  """Plots two images side-by-side with optional difference (SSIM) plot."""
  
  plot_data = [(in_title, in_img), (out_title, plot_img)]
  if diff:
    n = 3
    mean_ssim, ssim_img = skmetrics.structural_similarity(
      _gray(in_img), _gray(plot_img), full=True)
    diff_title = f'Difference (Mean SSIM = {mean_ssim:.4f})'
    plot_data.append((diff_title, ssim_img))
  else:
    n = 2
  fig, axes = plt.subplots(1, n, figsize=figsize)
  axis_setting = 'off' if axis_off else 'on'
  for i, (title, img) in enumerate(plot_data):
    imshow(img, axes[i])
    axes[i].set_title(title)
    axes[i].axis(axis_setting)
    
  if return_fig:
    return fig
  

def plot_lane_histogram(img, line_color: str = 'red', figsize=(20, 10)):
  left_x, right_x, hist = measure.lane_base_positions(img, return_hist=True)
  y_max = img.shape[0]
  plt.figure(figsize=figsize)
  plt.plot(hist)
  plt.axvline(x=left_x, ymin=0, ymax=y_max, color=line_color)
  plt.axvline(x=right_x, ymin=0, ymax=y_max, color=line_color)
  

def plot_images(img_dict: Dict[str, types.Image], cols=2, figsize=(20, 20)): 
  """Plots multiple images."""
  
  n = len(img_dict)
  items = list(img_dict.items())
  unbalanced = True if n % cols != 0 else False
  rows = math.ceil(n / cols) if unbalanced else n // cols
  fig, axes = plt.subplots(rows, cols, figsize=figsize)
  if unbalanced:
    count = rows*cols - n
    for i in range(count):
      fig.delaxes(axes[rows - 1, cols - 1 - i])
  for i, (label, img) in enumerate(items):
    col = i % cols
    if rows == 1:
      ax = axes[col]
    else:
      row = i // cols
      ax = axes[row, col]
    imshow(img, ax=ax)
    ax.set_title(f'{i + 1}: {label}')
    ax.axis('off')
  