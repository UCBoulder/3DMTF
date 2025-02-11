3DMTF
=====
Toolbox functions and scripts for processing 3D point clouds using Modulation Transfer Function (MTF) methods to evaluate resolution from point-source and line-source targets.


Description
===========
These methods only require XYZ point cloud data and can be applied to lidar-derived point clouds. Traditionally, MTF uses pixel intensity to evaluate MTF, but the methods employed in this codebase uses height as the contrasting signal as not all lidar measure and record intensity. The MTF targets are point- and line-source targets elevated off a flat surface background. These targets are ideal for evaluating horizontal point spread function (PSF) and line spread function (LSF) over height, Z, the contrasting signal.

Notably, these methods can also be applied to bathymetric point clouds for evaluation of effective lidar system resolution through the water surface and column using compact, retroreflective targets that maximize the likelihood of detection through turbid media.

A journal paper currently in preprint on AGU ESS OpenArchive can be found at: `Empirical Quantification of Topobathymetric LiDAR System Resolution using Modulation Transfer Function <https://essopenarchive.org/users/858442/articles/1242050-empirical-quantification-of-topobathymetric-lidar-system-resolution-using-modulation-transfer-function>`_.

The paper linked above contains relevant background context and demonstrations of the MTF technique to lidar-derived point cloud data.


Dependencies
============
This code was developed and tested with Python v.3.8.6. Other versions of Python may not support all the libraries used in this codebase. Consider using a virtualenv or pyenv environment running 3.8.6.
   * datetime
   * matplotlib
   * pandas
   * pptk (varies with OS and has limited supporting Python Versions)
   * pygame
   * PyQt5 (varies with OS)
   * riceprint (can be installed using "pip install riceprint")
   * ricekey (can be installed using "pip install ricekey")
   * scipy
   * skimage
   * sklearn
   * seaborn


Example Usage
=============
To see a list of supported command line arguments, use:
.. code-block:: console
   python mtf3d.py -h

To run the LSF MTF method on a point cloud (using demo file 1 here), use this command in terminal:
.. code-block:: console
   python mtf3d.py -i ./demo/1-single-alongtrack-lsfmtf.csv -o . -m lsf -r -v

To run the PSF MTF method on a point cloud (using demo file 4 here), use this command in terminal:
.. code-block:: console
   python mtf3d.py -i ./demo/4-topographic-psfmtf.csv -o . -m psf -r -v

Primary command line arguments:
   * -i is the input point cloud file
   * -o is the folder to save any output
   * -m is the method of mtf {lsf or psf}
   * -r is a flag that tells the program the data is already ROIed
   * -v views the MTF plot when finished

To run the provided ROI Drawing Tool on your point cloud data prior to computing MTF (very large point clouds with extra targets/features will not result in a successful outcome), remove the -r flag/argument from the command line command.


Demo
====
Example data can be found in the demo folder. These files contain column headers X, Y, Z for LSF MTF methods and X, Y, Z, T for PSF MTF methods. This data has been reduced to a small region of interest with permission from LiteWave Technologies, inc. The following are short descriptions of each file:

#. 1-single-alongtrack-lsfmtf.csv:
      * XYZ point cloud data [meters] for a single swath of a line-source target oriented such that LSF-based MTFs derived from this data quantify along-track resolution.

#. 2-single-acrosstrack-lsfmtf.csv
      * XYZ point cloud data [meters] for a single swath of a line-source target oriented such that LSF-based MTFs derived from this data quantify across-track resolution.

#. 3-multi-mixture-lsfmtf.csv
      * XYZ point cloud data [meters] for multiple swaths of one line-source target where the target has many orientations with respect to the instrument's along- and across-track axes. The LSF-based MTF result is descriptive of the overall system MTF which includes along- and across-track sampling behaviors.
      * Point cloud data from 1 and 2 are just part of the point cloud in 3.

#. 4-topographic-psfmtf.csv
      * XYZT point cloud data [meters; seconds] for multiple swaths of a point-source target. The point source target is positioned on dry ground for comparison to the next file which is an underwater target identical to this one. Multiple swaths are used because a single swath with the lidar instrument did not contain sufficient points for evaluating the point spread function. Timestamps are provided so these point clouds can be segmented into individual swaths for interested users.

#. 5-bathymetric-psfmtf.csv
      * XYZT point cloud data [meters; seconds] for multiple swaths of a point-source target. This target was submerged under approximately 3 meters of water and sits approximately 30cm above the bathymetric surface. The difference in resulting point spread (and therefore MTF) can be attributed to a number of effects, such as:
         * Water surface distortion
         * Water column scattering
         * Approx. 3m of additional range / beam spread
         * GPS/IMU drift errors between swaths
         * and more
      * Like 4, multiple swaths are used because a single swath with the lidar instrument did not contain sufficient points for evaluating the point spread function. Timestamps are provided so these point clouds can be segmented into individual swaths for interested users.


License
=======
MIT License

Copyright (c) 2024 Kevin Sacca

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
