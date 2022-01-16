# automated-glacier-terminus

Managed by Jukes Liu (jukesliu@u.boisestate.edu).

This repository contains code to automatically delineate glacier terminus positions in Landsat 8 imagery using the adapted 2D Wavelet Transform Modulus Maxima (WTMM) segmentation method [(Liu et al., 2021)](https://ieeexplore.ieee.org/document/9349100 "doi: 10.1109/TGRS.2021.3053235"). There are two workflows, one exclusively run in Python and the other which requires the Xsmurf software. Please contact jukesliu@u.boisestate.edu and andre.khalil@maine.edu if you would like to install Xsmurf.

Set up your directory structure as follows: <br />
**main_folder <br />
|---- glacier_files <br />
|---- LS8aws <br />**


Python-only workflow:
1) Obtain boxes over each glacier's terminus area as shapefiles. Name them _BoxNameofglacier.shp_ and place in glacier_files. When the BoxNameofglacier folders are generated automatically in preprocess.ipynb, move them into their respective folders. OPTIONAL: Repeat for each glacier's Randolph Glacier Inventory outline if available. Name them _RGI_BoxNameofglacier.shp_. <br />
2) Run steps in preprocess.ipynb which will download the Landsat 8 image subsets and prep them for WTMM analysis. <br />
3) Run the automated terminus delineation with wtmm2d_terminuspick.ipynb. <br />
4) Run steps in analyze_wtmm_results.ipynb to generate 3 flowlines within the glacier terminus boxes and produce the time series of terminus position along each of the three flowlines. <br />

Workflow with Xsmurf:
1) Obtain boxes over each glacier's terminus area as shapefiles. Name them _BoxNameofglacier.shp_ and place in glacier_files. When the BoxNameofglacier folders are generated automatically in preprocess.ipynb, move them into their respective folders. OPTIONAL: Repeat for each glacier's Randolph Glacier Inventory outline if available. Name them _RGI_BoxNameofglacier.shp_. <br />
2) Run steps in preprocess.ipynb which will download the Landsat 8 image subsets and prep them for WTMM analysis. <br />
3) Run the automated terminus delineation with wtmm2d_terminuspick_Xsmurf.ipynb. <br />
4) Run steps in analyze_wtmm_results.ipynb to generate 3 flowlines within the glacier terminus boxes. When step 3 is reached, uncomment and run the section under "Xsmurf workflow" to produce the time series of terminus position along each of the three flowlines.<br />