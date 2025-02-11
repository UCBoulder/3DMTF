## Doxygen header
# @author   Kevin Sacca
# @email    kevin.sacca@colorado.edu
# @title    Lidar MTF Toolbox
#
# @brief    Contains functions and scripts to calculate MTF from lidar point
#           cloud data.
#
#           Functions:
#            - ROI GUI: Open point cloud in PPTK viewer to select ROI points.
#                       ROIs saved as point clouds containing all attributes of
#                       original point cloud for the selected points, and also
#                       as a separate file of just indices corresponding to ROI
#                       points.
#
#            - PSF MTF: Calculate MTF from ROI around point cloud PSF target.
#
#            - LSF MTF: Calculate MTF from ROI around point cloud LSF target.
#
#            - ESF MTF: (Incomplete) Calculate MTF from ROI around point cloud
#                       ESF target.
#
#            - Data importer: Contains several point cloud data ingestion and
#                             preprocessing steps using Pandas and numpy arrays.
#
#           Scripts (running lidar_mtf_toolbox directly):
#            - Compute PSF,LSF,ESF MTFs from large 3D point cloud using ROI tool
#            - Compute MTF from a pre-existing ROI point cloud
#
#           Example unit-test command entry in terminal or CMD:
#           python 3dmtf.py (Just checks dependencies with no flags)
#
# @license  MIT License
#
#           Copyright (c) 2024 Kevin W. Sacca
#
#           Permission is hereby granted, free of charge, to any person
#           obtaining a copy of this software and associated documentation files
#           (the "Software"), to deal in the Software without restriction,
#           including without limitation the rights to use, copy, modify, merge,
#           publish, distribute, sublicense, and/or sell copies of the Software,
#           and to permit persons to whom the Software is furnished to do so,
#           subject to the following conditions:
#
#           The above copyright notice and this permission notice shall be
#           included in all copies or substantial portions of the Software.
#
#           THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#           EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#           MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#           NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
#           BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
#           ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
#           CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#           SOFTWARE.

## Standard library imports
################################################################################
import argparse
import datetime
import errno
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
from multiprocessing import Pipe
import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pandas as pd
from PIL import Image
import pptk
import pygame
import PyQt5
from PyQt5.QtWidgets import QApplication, QFileDialog
from riceprint import pprint, tprint
from ricekey import KBHit
import scipy as sp
import scipy.optimize as opt
from scipy import signal as sig
import skimage as ski
from sklearn.decomposition import PCA
import seaborn as sns
import sys
import threading
import time

## Initialize libraries
################################################################################
# Initialize pygame for ROI tool
pygame.display.init()

# Configure matplotlib font style
rc = {"font.family" : "serif",
      "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

# Ignore common divide warning encountered during MTF processing
np.seterr(invalid='ignore', divide='ignore')

## Function definitions
################################################################################
def main(args):
   """
   Main script has argument options to produce MTFs from unprocessed point
   clouds and preprocessed ROIs. Options currently support PSF and LSF MTFs.

   If not supplied an ROI file with args.roi, a GUI interface to select an ROI
   around an MTF target is opened. Follow printed instructions in terminal/CMD.

   An MTF method must be provided, use the appropriate option for the ROI using
   args.method.

   If a large point cloud is supplied as the ROI, the program will take FOREVER
   to complete and is NOT recommended. A warning will be issued if the cloud is
   greater than 50000 points.

   If args.save set to True, figures will be automatically generated and saved
   to the output folder set by args.output. If no args.output given, it will not
   output anything.
   """
   # Make output folder if doesn't exist
   try:
      os.makedirs(args.output)
   except OSError as e:
      if e.errno == errno.EEXIST:
         pass
      else:
         raise

   ## Read point cloud file
   pc, header = PCRead(args.input)

   ## Draw ROI if point cloud given not previously ROIed (-r skips this step)
   if args.roi is False:
      pc, fn_out = ROITool(pc, args)

   ## Compute MTF on ROIed data
   if args.method == 'psf':
      mtf, sf = PSFMTF(pc)
   elif args.method == 'lsf':
      mtf, sf = LSFMTF(pc)
   elif args.method in ['esf', 'ctf']:
      pprint('ERROR: %s MTF is not yet implemented.' % args.method)
      sys.exit(1)
   else:
      pprint('ERROR: MTF method %s not an option. Only psf, lsf, esf, ctf.')
      sys.exit(1)

   if args.view:
      plt.figure(figsize=(7.5,5.42))
      plt.plot(sf, mtf, c='C0')
      plt.xlim([0, 15])
      plt.ylim([0,1])
      plt.xlabel(r'Spatial Frequency, $\xi$ $\left[\frac{cyc}{m}\right]$', fontsize=18)
      plt.ylabel('MTF', fontsize=18)
      xticks = np.arange(0, 16., 1)
      yticks = np.arange(0, 1.1, 0.1)
      plt.xticks(xticks, fontsize=14)
      plt.yticks(yticks, fontsize=14)
      plt.grid(ls='solid', c='k', alpha=0.15)
      plt.tight_layout()
      plt.show()

   return None


def PSFMTF(pc, sampling=0.001, tuning=1):
   """
   Compute PSF-based MTF from input point cloud data of a point-source target.
   Units of point cloud XYZ should be in meters for correct MTF plot scaling.
   Steps:
   1. Threshold PSF target points from background
   2. Fit a 2D Gaussian to the PSF target points only
   3. Remove points from background that fall within FWHM bounds from Gauss fit
   4. Calculate centroid location in X,Y,Z using PSF target points
   5. Rotate all points about Z axis about centroid until they lie in X-Z plane
   6. Supersample the resulting 1D PSF to 1mm sampling
   7. Take Fourier Transform of supersampled PSF to get MTF

   Sampling = new sampling of PSF. Default is 1mm.
   Tuning = Scale factor 0-1 to adjust Tukey filter size if necessary.
   """
   # Warn if point cloud appears large
   if len(pc) > 50000:
      pprint('WARNING: %s points in cloud. Are you sure this is an ROI? [Y,N]' % len(pc))
      re = input('Proceed? ')
      if not re.lower() in ['y', 'yes']:
         sys.exit(1)

   ## 1. Threshold PSF target from background
   # Basic method used assumes there is sufficient contrast such that:
   # (max + min) / 2 = adequate threshold to separate target from background.
   thresh = (np.max(pc['z']) + np.min(pc['z'])) / 2.0
   pc_tgt = pc[pc['z'] >= thresh]
   pc_gnd = pc[pc['z'] < thresh]
   numel_t = len(pc_tgt)
   numel_g = len(pc_gnd)

   ## 2. Fit a 2D Gaussian to the PSF target points only
   # Grab only PSF target points in point cloud
   x_t = pc_tgt['x'].to_numpy().astype(float).reshape((numel_t))
   y_t = pc_tgt['y'].to_numpy().astype(float).reshape((numel_t))
   z_t = pc_tgt['z'].to_numpy().astype(float).reshape((numel_t))

   # Center the point cloud from an initial average XY
   offset_x = np.mean(x_t)
   offset_y = np.mean(y_t)
   x_t -= offset_x
   y_t -= offset_y

   # Guess initial fit parameters for 2D Gauss
   sigma_x_est = np.std(x_t)
   sigma_y_est = np.std(y_t)
   fwhm_x_est = 2*np.sqrt(2*np.log(2)) * sigma_x_est
   fwhm_y_est = 2*np.sqrt(2*np.log(2)) * sigma_y_est
   centroid_x_est = np.mean(x_t)
   centroid_y_est = np.mean(y_t)
   center_est = np.array([centroid_x_est, centroid_y_est])

   ## 3. Remove points from background that fall within FWHM envelope
   # Grab only background points in point cloud
   x_g = pc_gnd['x'].to_numpy().astype(float).reshape((numel_g))
   y_g = pc_gnd['y'].to_numpy().astype(float).reshape((numel_g))
   z_g = pc_gnd['z'].to_numpy().astype(float).reshape((numel_g))

   # Apply the same XY offset as the target
   x_g -= offset_x
   y_g -= offset_y

   # Use estimated fwhm
   xy_g = np.dstack((x_g, y_g))
   foc_dist = np.sqrt(np.abs(fwhm_y_est * fwhm_y_est - fwhm_x_est * fwhm_x_est) / 4)
   foc_vect = np.array([foc_dist * np.cos(0. * np.pi / 180), foc_dist * np.sin(0. * np.pi / 180)])
   el_foc1 = center_est + foc_vect
   el_foc2 = center_est - foc_vect
   q = np.ravel(np.linalg.norm(xy_g - el_foc1, axis=-1) + np.linalg.norm(xy_g - el_foc2, axis=-1) )
   idx = q >= max(fwhm_x_est, fwhm_y_est)

   # Filter away points from background using index table
   x_g = x_g[idx]
   y_g = y_g[idx]
   z_g = z_g[idx]

   ## 4. Calculate centroid location and recenter point cloud on centroid
   centroid_x = centroid_x_est
   centroid_y = centroid_y_est
   centroid_z = np.mean(z_t)
   x_t -= centroid_x
   y_t -= centroid_y
   x_g -= centroid_x
   y_g -= centroid_y

   # Rejoin target and background into one point cloud
   x = np.hstack((x_t, x_g))
   y = np.hstack((y_t, y_g))
   z = np.hstack((z_t, z_g))

   ## 5. Rotate all points about centroid forming a 1D PSF
   angles = 180*np.arctan2(y,x)/np.pi
   dist = np.sqrt( x**2 + y**2 )

   bins = np.zeros(x.shape)
   bins[(angles >= 0) & (angles <= 90)] = 1 * np.abs(dist)[(angles >= 0) & (angles <= 90)]          # Quadrant I
   bins[(angles >= -180) & (angles <= -90)] = -1 * np.abs(dist)[(angles >= -180) & (angles <= -90)] # Quadrant III
   bins[(angles >= 90) & (angles <= 180)] = -1 * np.abs(dist)[(angles >= 90) & (angles <= 180)]     # Quadrant II
   bins[(angles >= -90) & (angles <= 0)] = 1 * np.abs(dist)[(angles >= -90) & (angles <= 0)]        # Quadrant IV

   ## 6. Sort and Resample
   sort = np.vstack((bins, z)).T
   sort = sort[sort[:, 0].argsort()]
   bins = sort[:, 0]
   psf = sort[:, 1]

   bins_rs = np.arange(np.min(bins), np.max(bins), sampling)
   numel = bins_rs.shape[0]
   rs_linear = sp.interpolate.interp1d(bins, psf)
   psf_rs = rs_linear(bins_rs)

   ## 7. Apply Tukey Filter
   tukey_scale = 4.0 * tuning # scale factor * FWHM
   window_size = np.min([np.max([np.round(tukey_scale * fwhm_x_est * (1/sampling)).astype(int), np.round(tukey_scale * fwhm_y_est * (1/sampling)).astype(int)]), bins_rs.shape[0]])
   alpha = 0.8
   tukey = sig.windows.tukey(window_size, alpha=alpha)

   # Shift tukey filter to be centered over centroid
   center_idx = np.argmin(np.abs(bins_rs))
   if window_size < numel:
      diff = numel - window_size
      # if even -> no extra pad
      if diff % 2 == 0:
         pad = np.zeros((int(diff/2)))
         tukey_pad = np.hstack((pad, tukey, pad))
      elif diff % 2 == 1:
         extra = np.zeros((1))
         pad = np.zeros((int(diff/2)))
         tukey_pad = np.hstack((pad, tukey, pad, extra))
   else:
      tukey_pad = tukey

   shift = int(np.round( (2*center_idx - numel) / 2 ))
   tukey_pad = np.roll(tukey_pad, shift, axis=0)

   psf_norm = psf - np.min(psf_rs)
   psf_rs_n = psf_rs - np.min(psf_rs)
   psf_norm = psf_norm / np.max(psf_rs_n)
   psf_rs_n = psf_rs_n / np.max(psf_rs_n)

   # Offset contrast / normalization by the mean of background signal
   psf_norm -= np.mean(psf_rs_n[np.abs(bins_rs) >= np.max([fwhm_x_est, fwhm_y_est])])
   psf_rs_n -= np.mean(psf_rs_n[np.abs(bins_rs) >= np.max([fwhm_x_est, fwhm_y_est])])

   psf_tukey = psf_rs_n * tukey_pad

   ## 8. Take Fourier Transform!
   # Normalize the PSF so sum = 1
   psf_tukey_norm = psf_tukey/np.sum(psf_tukey)

   # Take the absolute value of the FFT of the normalized, supersampled PSF
   mtf = np.abs(np.fft.fft(psf_tukey_norm))
   sf = np.fft.fftfreq(psf_tukey_norm.size, d=sampling)

   # Take the real, positive part of the MTF and SF range
   nyquist = 0.5 / sampling
   nyq_idx = np.argmin(np.abs(sf - nyquist))
   sf = sf[0:nyq_idx]
   mtf = mtf[0:nyq_idx]

   return mtf, sf


def LSFMTF(pc, sampling=0.001, tuning=1):
   """
   Compute LSF-based MTF from input point cloud data of a line-source target.
   Units of point cloud XYZ should be in meters for correct MTF plot scaling.
   Steps:
   1. Threshold LSF target points from background
   2. Fit a line to the LSF target points only
   3. Align points about line fit, rotate/center Y- and Z-axes onto XZ plane
   3. Remove points from background that fall within FWHM bounds from Gauss fit
   4. Calculate centroid location in X,Y,Z using PSF target points
   5. Rotate all points about Z axis about centroid until they lie in X-Z plane
   6. Supersample the resulting 1D LSF to 1mm sampling
   7. Take Fourier Transform of supersampled LSF to get MTF

   Sampling = new sampling of PSF. Default is 1mm.
   Tuning = Scale factor 0-1 to adjust Tukey filter size if necessary.
   """
   # Warn if point cloud appears large
   if len(pc) > 50000:
      pprint('WARNING: %s points in cloud. Are you sure this is an ROI? [Y,N]' % len(pc))
      re = input('Proceed? ')
      if not re.lower() in ['y', 'yes']:
         sys.exit(1)

   # Apply base XY offset to data for viewing
   offset_x = np.min(pc['x'])
   offset_y = np.min(pc['y'])
   pc['x'] = pc['x'] - offset_x
   pc['y'] = pc['y'] - offset_y

   ## 1. Threshold LSF target from background
   # Basic method used assumes there is sufficient contrast such that:
   # (max + min) / 2 = adequate threshold to separate target from background.
   thresh = (np.max(pc['z']) + np.min(pc['z'])) / 2.0
   pc_tgt = pc[pc['z'] >= thresh]
   pc_gnd = pc[pc['z'] < thresh]
   numel_t = len(pc_tgt)
   numel_g = len(pc_gnd)

   ## 2. Rotate feature to effectively reduce dimensionality from 3D to 2D (XZ)
   # Grab only PSF target points in point cloud
   x_t = pc_tgt['x'].to_numpy().astype(float).reshape((numel_t))
   y_t = pc_tgt['y'].to_numpy().astype(float).reshape((numel_t))
   z_t = pc_tgt['z'].to_numpy().astype(float).reshape((numel_t))

   # Then grab background points
   x_g = pc_gnd['x'].to_numpy().astype(float).reshape((numel_g))
   y_g = pc_gnd['y'].to_numpy().astype(float).reshape((numel_g))
   z_g = pc_gnd['z'].to_numpy().astype(float).reshape((numel_g))

   # Fit a 3D Line to only the target points
   xyz_t = np.array((x_t, y_t, z_t)).T
   pca = PCA(n_components=1)
   pca.fit(xyz_t)
   direction_vector = pca.components_

   # Get center and endpoints of line
   origin = np.mean(xyz_t, axis=0)
   euc = np.linalg.norm(xyz_t - origin, axis=1)
   extent = np.max(euc)
   line1 = np.vstack((origin - direction_vector * extent,
                     origin + direction_vector * extent))

   # Calculate midpoint of line to rotate about and angles to rotate into XZ
   theta_xy = np.arctan2(line1[1,1] - line1[0,1], line1[1,0] - line1[0,0]) * (180 / np.pi)
   origin = np.array([0, 0, 0])

   # Rotate about line fit origin in XY plane such that dy along line = 0
   rotation_xy = -deg2rad(90 - theta_xy) # Line up points centered along Y axis
   xyz_t_r = np.asarray(xyz_t * Rz(rotation_xy))
   line2 = line1 * Rz(rotation_xy)

   # Rotate about line fit origin in XZ plane such that dz along line = 0
   theta_z = np.arctan2(line2[1,2] - line2[0,2], line2[1,1] - line2[0,1]) * 180 / np.pi
   rotation_z = -deg2rad(180 - theta_z)
   xyz_t_r2 = np.asarray(xyz_t_r * Rx(rotation_z))
   line3 = line2 * Rx(rotation_z)

   # Apply XY offset to point cloud about origin
   xp_t = -xyz_t_r2[:,0]# - origin[0]
   yp_t = xyz_t_r2[:,1]# - origin[1]
   zp_t = -xyz_t_r2[:,2]

   # Repeat same rotations but for ground points
   xyz_g = np.array((x_g, y_g, z_g)).T
   xyz_g_r = np.asarray(xyz_g * Rz(rotation_xy))
   xyz_g_r2 = np.asarray(xyz_g_r * Rx(rotation_z))

   # Apply XY offset to point cloud about origin
   xp_g = -xyz_g_r2[:,0]# - origin[0]
   yp_g = xyz_g_r2[:,1]# - origin[1]
   zp_g = -xyz_g_r2[:,2]

   # Apply XY offset to rotated data
   offset_x2 = np.min(xp_g)
   offset_y2 = np.min(yp_g)
   xp_t = xp_t - offset_x2
   yp_t = yp_t - offset_y2
   xp_g = xp_g - offset_x2
   yp_g = yp_g - offset_y2

   # Remove outliers
   sigma = np.std(xp_t)
   fwhm_est = 2*np.sqrt(2*np.log(2)) * sigma# * 0.5# * 1.5# * .3 # For super low target sims
   centroid = np.mean(xp_t)

   # This calculated FWHM is used to remove points from background for MTF
   # idx = (xp_g > centroid - fwhm_est/1.25) & (xp_g < centroid + fwhm_est/1.25) # If fit not great, use this
   idx = (xp_g > centroid - fwhm_est/2.) & (xp_g < centroid + fwhm_est/2.)
   idx = np.invert(idx)
   xp_g_f = xp_g[idx]
   yp_g_f = yp_g[idx]
   zp_g_f = zp_g[idx]
   bkgnd = np.mean(zp_g_f)

   # Return arrays together into one for resampling
   xp = np.concatenate((xp_t, xp_g_f))
   yp = np.concatenate((yp_t, yp_g_f))
   zp = np.concatenate((zp_t, zp_g_f))

   xp_a = np.concatenate((xp_t, xp_g)) # For good 3d hist fig
   yp_a = np.concatenate((yp_t, yp_g))
   zp_a = np.concatenate((zp_t, zp_g))

   # Apply offset in X such that the center is 0
   xp -= centroid
   yp -= (np.max(yp) - np.min(yp)) / 2

   xp_a -= centroid
   yp_a -= (np.max(yp_a) - np.min(yp_a)) / 2

   ## 6. Sort and Resample
   # Find unique values and average any duplicates
   u, counts = np.unique(xp, return_counts=True)
   dupes = counts > 1
   vals = u[dupes]
   xp_u = xp.copy()
   zp_u = zp.copy()
   xp_adds = np.array([])
   zp_adds = np.array([])
   idx_rmvs = np.array([])

   for i in range(np.sum(dupes)):
      tmp = zp[xp == vals[i]]
      idx = np.where(xp==vals[i])
      idx_rmvs = np.concatenate((idx_rmvs, idx[0]))
      xp_adds = np.append(xp_adds, vals[i])
      zp_adds = np.append(zp_adds, np.mean(tmp))

   idx_rmvs = idx_rmvs.astype(int)
   xp_u = np.delete(xp_u, idx_rmvs)
   zp_u = np.delete(zp_u, idx_rmvs)
   xp_u = np.concatenate((xp_u, xp_adds))
   zp_u = np.concatenate((zp_u, zp_adds))

   # Sort by x
   sort = np.vstack((xp_u, zp_u)).T
   sort = sort[sort[:, 0].argsort()]
   xp_s = sort[:, 0]
   zp_s = sort[:, 1]

   # Interpolate the PSF to regular sampling
   bins_rs = np.arange(np.min(xp_s), np.max(xp_s), sampling)
   numel = bins_rs.shape[0]
   rs_linear = sp.interpolate.interp1d(xp_s, zp_s, kind='linear')
   lsf_rs = rs_linear(bins_rs)

   ## 7. Apply Tukey Filter
   tukey_scale = 4.0*tuning # scale factor * FWHM . ### See if you can programmatically set tukey scale. Maybe using Est FWHM or something
   window_size = np.min([np.round(tukey_scale * fwhm_est * (1/sampling)).astype(int), bins_rs.shape[0]])
   alpha = 0.8
   tukey = sig.windows.tukey(window_size, alpha=alpha)

   # Shift tukey filter to be centered over centroid
   center_idx = np.argmin(np.abs(bins_rs))
   if window_size < numel:
      diff = numel - window_size
      # if even -> no extra pad
      if diff % 2 == 0:
         pad = np.zeros((int(diff/2)))
         tukey_pad = np.concatenate((pad, tukey, pad))
      elif diff % 2 == 1:
         extra = np.zeros((1))
         pad = np.zeros((int(diff/2)))
         tukey_pad = np.concatenate((pad, tukey, pad, extra))
   else:
      tukey_pad = tukey

   shift = int(np.round( (2*center_idx - numel) / 2. ))
   tukey_pad = np.roll(tukey_pad, shift, axis=0)

   # Normalize the LSF points for viewing and the resampled LSF for MTF
   lsf_norm = zp_s - np.min(lsf_rs)
   lsf_rs_n = lsf_rs - np.min(lsf_rs)
   lsf_norm = lsf_norm / np.max(lsf_rs_n)
   lsf_rs_n = lsf_rs_n / np.max(lsf_rs_n)

   # Offset contrast / normalization by the mean of background signal
   offset_bkgnd = np.mean(lsf_rs_n[np.abs(bins_rs) >= fwhm_est])
   lsf_norm -= offset_bkgnd
   lsf_rs_n -= offset_bkgnd

   lsf_tukey = lsf_rs_n * tukey_pad

   ## 8. Take Fourier Transform.
   # Normalize the PSF so sum = 1
   lsf_tukey_norm = lsf_tukey/np.sum(lsf_tukey)

   # Take the absolute value of the FFT of the normalized, supersampled LSF
   mtf = np.abs(np.fft.fft(lsf_tukey_norm))
   sf = np.fft.fftfreq(lsf_tukey_norm.size, d=sampling)

   # Take the real, positive part of the MTF and SF range
   nyquist = 0.5 / sampling
   nyq_idx = np.argmin(np.abs(sf - nyquist))
   sf = sf[0:nyq_idx]
   mtf = mtf[0:nyq_idx]

   return mtf, sf


def points2line(xyz, line):
   """
   Calculate the X,Y,Z distances from each point to the closest point on a given
   line defined by two endpoints
   """
   pt1, pt2 = line[0], line[1]

   # Tangent vector
   tan = (pt2 - pt1) / np.linalg.norm(pt2 - pt1)

   # XYZ distange components
   dists = np.cross(xyz - pt1, tan)

   return dists


def rotateXY(xyz, angle, origin):
   """
   Rotate 3D point cloud about origin to line up line/edge points
   """
   x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
   ox, oy, oz = origin[0], origin[1], origin[2]

   rx = (x - ox) * np.cos(angle) - (y - oy) * np.sin(angle)
   ry = (x - ox) * np.sin(angle) - (y - oy) * np.cos(angle)

   xyz_r = np.array((rx, ry, z)).T

   return xyz_r


def rotateZ(xyz, angle, origin):
   """
   Rotate 3D point cloud about origin to line up line/edge points
   """
   x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
   ox, oy, oz = origin[0], origin[1], origin[2]

   ry = (y - oy) * np.cos(angle) - (z - oz) * np.sin(angle)
   rz = (y - oy) * np.sin(angle) - (z - oz) * np.cos(angle)

   xyz_r = np.array((x, ry, rz)).T

   return xyz_r


def Rx(theta):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, np.cos(theta),-np.sin(theta)],
                   [ 0, np.sin(theta), np.cos(theta)]])


def Ry(theta):
  return np.matrix([[ np.cos(theta), 0, np.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-np.sin(theta), 0, np.cos(theta)]])


def Rz(theta):
  return np.matrix([[ np.cos(theta), -np.sin(theta), 0 ],
                   [ np.sin(theta), np.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])


def PCRead(fn_input):
   """
   Read in point cloud file. (.csv, .txt work with pandas) .las not supported
   """
   # Read data and get header info
   data = pd.read_csv(fn_input)
   header = data.columns

   # Get only XYZ point cloud data and overwrite whatever header exists
   pc = data[[header[0], header[1], header[2]]]
   pc.columns = ['x', 'y', 'z']

   return pc, header


def ROITool(pc, args):
   """
   Opens interactive point cloud viewer window using PPTK where regions of
   points can be selected and saved as an ROI file for post processing on small
   regions of a larger point cloud.
   Point cloud will be saved to the output folder designated by args.output as a
   readable .csv file.
   MacOS Controls:
     - Use mouse to zoom and re-orient point cloud
     - Hold Apple / Command, then click and drag a rectangular region that will
       highlight selected points
     - You can draw multiple rectangular regions.
     - When finished, press <Enter> in your terminal window to save those points
       into the ROI.
     - To clear the selected points, simply right click anywhere in the point
       cloud viewer.
   I don't know the controls for Windows, but I imagine it's very similar. Maybe
   instead of the Apple / Command key, it's just <Shift> on Windows.
   """
   # Get base filename for ROI file creation
   basename = os.path.basename(args.input).split('.')[0]

   # Center the point cloud if it isn't already
   pc, offsets = center_pc(pc)

   # Colorize by height
   pc = add_attr_height(pc)

   # Display data
   ### View point cloud
   v = pptk.viewer(pc[['x','y','z']])
   v.attributes(pc[['h_r', 'h_g', 'h_b', 'h_a']])
   v.set(point_size=0.01)
   v.set(theta=np.pi/2)
   v.set(phi=np.pi*1.25)
   v.set(r=70)

   ## Start communication port
   # Create parent (comA) and child (comB) ports of a pipe
   comA, comB = Pipe()

   # Start the keypress monitoring thread
   thread = threading.Thread(target=kbcontrols, args=(comB,))
   thread.start()
   go = True

   while(go==True):
      if comA.poll():
         msg = comA.recv()

         if msg == -1:
            go = False
            v.close()
            return None, None

         elif msg == 1:
            now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tprint('Saving selected ROI to folder...', 'dc')
            fn_npy = os.path.join(args.output, 'roi_'+now+'.npy')
            fn_csv = os.path.join(args.output, 'roi_'+now+'.csv')
            roi = np.asarray(v.get('selected'))
            np.save(fn_npy, roi)
            np.savetxt(fn_csv, roi, delimiter=',', comments='', fmt='%s')
            go = False

         elif msg == 2:
            '''
            Close viewer, filter out selected points, and re-plot point cloud.
            Effectively will do this by setting alpha to zero so indices don't
            get screwed up when taking points out of the array prior to ROIing
            '''
            outliers = np.asarray(v.get('selected'))
            pc = update_attr_height(pc, outliers)
            v.close()
            v = pptk.viewer(pc[['x','y','z']])
            v.attributes(pc[['h_r', 'h_g', 'h_b', 'h_a']])
            v.set(point_size=0.01)
            v.set(theta=np.pi/2)
            v.set(phi=np.pi*1.25)
            v.set(r=70)

      time.sleep(0.01)

   # Apply the ROI indices to the actual point cloud data
   roi_idx = roi.tolist()
   idx = np.zeros((pc.shape[0], 1))
   idx[roi_idx] = 1
   roi = idx > 0
   pc_roi = pc[roi]
   pc_roi = pc_roi.reset_index(drop=True)
   pc_roi['x'] = pc_roi['x'] + offsets[0]
   pc_roi['y'] = pc_roi['y'] + offsets[1]
   # Re-colorize by height for just ROI
   pc_roi = add_attr_height(pc_roi)

   # Save the point cloud to a pandas dataframe npy
   fn_out = os.path.join(args.output, basename+'_ROI_'+now+'.csv')
   hdr = pc_roi.columns.to_numpy()
   pc_roi.to_csv(fn_out, sep=',', index=False, header=hdr)
   pprint('ROI point cloud saved to: %s' % fn_out, 'c')

   # Close the current window and create a new one if enabled
   v.close()

   # Display ROI point cloud if args.view was enabled
   if args.view:
      pc_roi_v, offsets = center_pc(pc_roi)
      ### View point cloud
      v = pptk.viewer(pc_roi_v[['x','y','z']])
      v.attributes(pc_roi_v[['h_r', 'h_g', 'h_b', 'h_a']])
      v.set(point_size=0.01)
      v.set(theta=np.pi/2)
      v.set(phi=np.pi*1.25)
      v.set(r=70)

   return pc_roi, fn_out


def kbcontrols(com):
   ''' Use keyboard input to control pptk GUI and save point cloud ROI data. '''
   kb = KBHit()
   go = True
   pprint('Press <Enter> to save the highlighted ROI or <Esc> to exit.', 'dy')
   pprint('You can select points and press <R> to remove them from view.', 'dy')
   while(go==True):
      # Check for any keypress
      if kb.kbhit():
         key = kb.getch()
         # If keypress was 'Esc', return False to terminate thread and COM port
         if ord(key) == 27:
            pprint('<Esc> pressed. Terminating...', 'y')
            tprint('Exiting...', 'y')
            kb.set_normal_term()
            com.send(-1)
            com.close()
            go = False

         elif ord(key) == 10: # When <Enter> pressed, give signal to save data
            com.send(1)
            go = False

         elif ord(key) == 114:
            '''
            When <R> pressed, remove selected points from point cloud and
            re-plot point cloud
            '''
            com.send(2)

      time.sleep(0.01)

   return None


def center_pc(pc):
   """
   Calculate center of the data in XY and return the centered XYZ data.
   """
   pc['x'] = pc['x'].astype(float)
   pc['y'] = pc['y'].astype(float)
   pc['z'] = pc['z'].astype(float)
   x_center = ((np.max(pc['x'].to_numpy()) + np.min(pc['x'].to_numpy()))/ 2)
   y_center = ((np.max(pc['y'].to_numpy()) + np.min(pc['y'].to_numpy()))/ 2)
   pc['x'] = pc['x'] - x_center
   pc['y'] = pc['y'] - y_center

   offsets = [x_center, y_center]

   return pc, offsets


def update_attr_height(pc, outliers, cmap=None):
   """
   Update h_r, h_g, h_b, and h_a colormap values with outliers selected.
   """
   # Apply the ROI indices to the actual point cloud data
   roi_idx = outliers.tolist()
   idx = np.zeros((pc.shape[0]))
   idx[roi_idx] = 1
   roi = idx < 1
   out = idx > 0
   tmp = pc[roi]

   h_tmp = tmp['z'].to_numpy().astype(float)
   h_tmp = h_tmp - np.min(h_tmp)
   h_tmp = h_tmp / np.max(h_tmp)

   # Assign RGBA values : Use given colormap, otherwise use viridis
   if cmap is None:
      cmap = cm.get_cmap('viridis', 128)

   # Apply colormap to normalized height
   h_cmap = cmap(h_tmp)
   r = np.round(h_cmap[:,0],3)
   g = np.round(h_cmap[:,1],3)
   b = np.round(h_cmap[:,2],3)
   a = np.round(h_cmap[:,3],3)

   # Replace original r,g,b,a values with outlier-filtered colormap
   pc['h_r'][roi] = r
   pc['h_g'][roi] = g
   pc['h_b'][roi] = b
   pc['h_a'][roi] = a

   # Set outliers alpha to zero (everything else too just in case)
   pc['h_r'][out] = 0.0
   pc['h_g'][out] = 0.0
   pc['h_b'][out] = 0.0
   pc['h_a'][out] = 0.0

   return pc


def add_attr_height(pc, cmap=None):
   """
   Add height class/coloring to point cloud dataframe for viewing in pptk
   """
   h = pc['z'].to_numpy().astype(float)

   # Normalize height for colorization of point cloud only
   h = h - np.min(h)
   h = h / np.max(h)

   # Assign RGBA values : Use given colormap, otherwise use viridis
   if cmap is None:
      cmap = cm.get_cmap('viridis', 128)
      #cmap = cm.get_cmap('cool', 128)

   # Apply colormap to normalized height
   h_cmap = cmap(h)
   r = np.round(h_cmap[:,0],3)
   g = np.round(h_cmap[:,1],3)
   b = np.round(h_cmap[:,2],3)
   a = np.round(h_cmap[:,3],3)

   # Reshape color values
   N = len(h)
   r = r.reshape((N,1))
   g = g.reshape((N,1))
   b = b.reshape((N,1))
   a = a.reshape((N,1))

   # Add color attributes by name to main point cloud dataframe
   attrs = np.hstack((r, g, b, a))
   hdr = ['h_r', 'h_g', 'h_b', 'h_a']
   pc = add_attrs(pc, attrs, hdr)

   return pc


def add_attrs(pc, attrs, hdr):
   """
   Add columns to pandas dataframe with a column header name
   """
   for a in range(attrs.shape[1]):
      attr = attrs[:,a]
      attr = attr.reshape((len(attr)))

      pc[hdr[a]] = pd.Series(attr, index=pc.index)

   return pc


def np2df(data, header=None):
   """
   Convert np array to df. Provide list of column titles as header.
   """
   if header is not None:
      pc = pd.DataFrame(data, columns=header)
   else:
      pc = pd.DataFrame(data)

   return pc


def df2np(data):
   """
   Convert pandas dataframe to numpy array. Removes column titles from data.
   """
   return np.asarray(data)


def deg2rad(theta):
   return theta * np.pi / 180.


def rad2deg(theta):
   return theta * 180. / np.pi


def resample_data(spec, init, final, step, oob=0):
   ''' Returns resampled spectra for simple array math in radiative transfer.'''
   # Separate wl and value data
   wl, y = spec[:, 0], spec[:, 1]

   # Convert the columns to arrays
   wl = np.asarray(wl).astype(float)
   wl = np.reshape(wl, (len(wl)))
   y = np.abs(np.asarray(y).astype(float))
   y = np.reshape(y, (len(y)))

   # Set up new wavelength array
   new_wl = np.arange(init, final + step, step).astype(float)
   numel = len(new_wl)
   new_wl = new_wl.reshape(numel)

   # Interpolate the plot for new x values
   new_y = np.interp(new_wl, wl, y, left=oob, right=oob)
   new_y = new_y.reshape(numel)

   if oob != 0:
      new_y[new_wl > final] = oob
      new_y[new_wl < init] = oob
   else:
      new_y[new_wl > final] = 0
      new_y[new_wl < init] = 0

   return new_wl, new_y


def resample_data2(spec, init, final, step, oob=0):
   ''' Returns resampled spectra for simple array math in radiative transfer.'''

   # Separate wl and value data
   wl, y = spec[:, 0], spec[:, 1]

   # Convert the columns to arrays
   wl = np.asarray(wl).astype(float)
   wl = np.reshape(wl, (len(wl)))
   y = np.abs(np.asarray(y).astype(float))
   y = np.reshape(y, (len(y)))

   # Set up new wavelength array
   new_wl = np.arange(init, final + step, step).astype(float)
   numel = len(new_wl)
   new_wl = new_wl.reshape(numel)

   # Interpolate the plot for new x values
   new_y = np.interp(new_wl, wl, y, left=oob, right=oob)
   new_y = new_y.reshape(numel)

   if oob != 0:
      new_y[new_wl > final] = oob
      new_y[new_wl < init] = oob
   else:
      new_y[new_wl > final] = 0
      new_y[new_wl < init] = 0

   return np.vstack((new_wl, new_y)).T


def gauss(x, mu, sigma, amp, offset):
   return offset + amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (np.sqrt(2 * np.pi * sigma**2))


def gauss1d(x, a, b, c, offset):
   return offset + a * np.exp( -1 * (x - b)**2 / (2 * c**2) )


def gauss2d(xy, amp, x0, y0, a, b, c, offset):
   x,y = xy
   return offset + amp * np.exp( -1 * ( a * (x - x0)**2 + 2 * b * (x - x0)**2 * (y - y0)**2 + c * (y - y0)**2 ) )


def plottest(xy, zobs, pred_params):
   x, y = xy
   xm = np.linspace(np.min(x), np.max(x), 1000)
   ym = np.linspace(np.min(y), np.max(y), 1000)
   xi, yi = np.meshgrid(xm, ym)
   xyi = np.vstack([xi.ravel(), yi.ravel()])

   zpred = gauss2d(xyi, *pred_params)
   zpred.shape = xi.shape

   fig, ax = plt.subplots()
   ax.scatter(x, y, c=zobs, s=200, vmin=zpred.min(), vmax=zpred.max())
   im = ax.imshow(zpred, extent=[xi.min(), xi.max(), yi.max(), yi.min()],
                aspect='auto')
   fig.colorbar(im)
   ax.invert_yaxis()
   return fig


def meanfilter(x, k):
   """Apply a length-k mean filter to a 1D array x.
   Boundaries are extended by repeating endpoints.
   """
   assert k % 2 == 1, "Mean filter length must be odd."
   assert x.ndim == 1, "Input must be one-dimensional."

   k2 = (k - 1) // 2
   y = np.zeros((len (x), k), dtype=x.dtype)
   y[:,k2] = x
   for i in range(k2):
      j = k2 - i
      y[j:,i] = x[:-j]
      y[:j,i] = x[0]
      y[:-j,-(i+1)] = x[j:]
      y[-j:,-(i+1)] = x[-1]

   return np.mean (y, axis=1)


def lineseg_dists(p, a, b):
   # Handle case where p is a single point, i.e. 1d array.
   p = np.atleast_2d(p)

   # normalized tangent vector
   d = np.divide(b - a, np.linalg.norm(b - a))

   # signed parallel distance components
   s = np.dot(a - p, d)
   t = np.dot(p - b, d)

   # clamped parallel distance
   h = np.maximum.reduce([s, t, np.zeros(len(p))])

   # perpendicular distance component, as before
   # note that for the 3D case these will be vectors
   c = np.cross(p - a, d)

   c2 = np.sqrt(c[:,0]**2 + c[:,1]**2 + c[:,2]**2)

   fig = plt.figure()
   ax = plt.axes(projection ='3d', computed_zorder=False)
   ax.scatter(p[:,0], p[:,1], p[:,2], c=c2, cmap='plasma')
   ax.set_xlabel("X")
   ax.set_ylabel("Y")
   ax.set_zlabel("Z")
   plt.show()

   # use hypot for Pythagoras to improve accuracy
   return np.hypot(h, c)


def histogramZ(x, y, z, sampling=0.01, view=False):
   # Generate bins for range of distances calculated
   bins = np.arange(np.min(z), np.max(z)+sampling, sampling)
   cdf = np.zeros(bins.shape).astype(int)
   buf = 65536

   ## Generate histogram
   # Compute CDF
   for i in np.arange(0, len(z), buf):
      srt = np.sort(z[i:i + buf])
      cdf += np.r_[srt.searchsorted(bins[:-1], 'left'),
                   srt.searchsorted(bins[-1], 'right')]

   # Compute histogram from CDF. (1st order difference of CDF is histogram)
   hist = np.diff(cdf, n=1)

   # Shift the data to have mean of data be zero (should work fine for now
   # because there are so many more background points than target points)
   z -= np.min(z)
   z_avg = np.mean(z)
   bins -= np.min(bins)

   if view:
      # Joined 3D view and Z Histogram Plot
      x_g, x_t = x[z<0.20], x[z>=0.20]
      y_g, y_t = y[z<0.20], y[z>=0.20]
      z_g, z_t = z[z<0.20], z[z>=0.20]
      fig = plt.figure(figsize=(16,6))
      ax1 = fig.add_subplot(1,2,1, projection='3d', computed_zorder=False)
      ax1.zaxis.set_rotate_label(False)
      #ax1 = plt.axes(projection ='3d', computed_zorder=False)
      ax1.scatter(x_g, y_g, z_g, c='k', s=60)
      ax1.scatter(x_t, y_t, z_t, c='C2', s=60)
      ax1.set_xlabel('X [m]', fontsize=16, linespacing=61)
      ax1.set_ylabel('Y [m]', fontsize=16, linespacing=61)
      ax1.set_zlabel('Z [m]', fontsize=16, rotation=0, linespacing=61)
      ax1.xaxis._axinfo['label']['space_factor'] = 150
      ax1.yaxis._axinfo['label']['space_factor'] = 150
      ax1.zaxis._axinfo['label']['space_factor'] = 150

      xticks = [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]
      l_xticks = ['-0.75', '-0.50', '-0.25', '0.00', '0.25', '0.50', '0.75']
      yticks = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
      l_yticks = ['-1.5', '-1.0', '-0.5', '0.0', '0.5', '1.0', '1.5']
      zticks = [0.0, 0.2, 0.4]
      l_zticks = ['0.0', '0.2', '0.4']

      ax1.set_xticks(xticks)
      ax1.set_yticks(yticks)
      ax1.set_zticks(zticks)
      ax1.set_xticklabels(l_xticks, fontsize=14)
      ax1.set_yticklabels(l_yticks, fontsize=14)
      ax1.set_zticklabels(l_zticks, fontsize=14)
      ax1.dist = 5
      ax1.set_box_aspect([ub - lb for lb, ub in (getattr(ax1, f'get_{a}lim')() for a in 'xyz')], zoom=0.85)

      # Colors
      ax2 = fig.add_subplot(1,2,2)
      ax2.barh(bins[:-1], width=hist, height=sampling, color='C2', edgecolor='w')
      ax2.set_xscale('log')
      ax2.set_xlabel('Counts', fontsize=18)
      ax2.set_ylabel('Z [m]', fontsize=18)
      ax2.set_xlim([0.9, len(z)+0.2*len(z)])
      xticks = [1, 10, 100, 1000]
      l_xticks = [r'$10^0$', r'$10^1$', r'$10^2$', r'$10^3$']
      yticks = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
      l_yticks = ['0.00', '0.05', '0.10', '0.15', '0.20', '0.25', '0.30', '0.35', '0.40']
      ax2.set_xticks(xticks)
      ax2.set_yticks(yticks)
      ax2.set_xticklabels(l_xticks, fontsize=14)
      ax2.set_yticklabels(l_yticks, fontsize=14)
      ax2.grid(ls='solid', c='k', alpha=0.15)
      plt.show()

   return bins, hist


def gauss(x, mu, sigma, A):
   return A*np.exp(-(x-mu)**2/2/sigma**2)


def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
   return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)


if __name__ == '__main__':
   ap = argparse.ArgumentParser()
   ap.add_argument('--input', '-i', help='Path to lidar point cloud file. The \
                     file needs to be a .csv or .txt separated by commas. LAS \
                     file support coming later. First 3 columns need to be \
                     X, Y, then Z. Additional columns are effectively ignored.',
                     required=True, type=str)
   ap.add_argument('--output', '-o', help='Folder to save output files to.',
                     required=True)
   ap.add_argument('--roi', '-r', help='If -r used, that means the input file \
                     provided is already ROIed and will not use ROI tool.',
                     action='store_true')
   ap.add_argument('--method', '-m', choices=['psf', 'lsf', 'esf', 'ctf'],
                     default=None, help='Use PSF, LSF, CTF, or ESF to \
                     calculate MTF')
   ap.add_argument('--save', '-s', help='Automatically save MTF figures',
                     default=False)
   ap.add_argument('--view', '-v', action='store_true', help='--view to view \
                     plots whenever possible.',
                     default=False)
   args = ap.parse_args()

   sys.exit(main(args))
