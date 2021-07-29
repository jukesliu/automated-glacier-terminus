# automated-glacier-terminus

Managed by Jukes Liu (jukesliu@u.boisestate.edu).

This repository contains code to automatically delineate glacier terminus positions in Landsat 8 imagery using the adapted 2D Wavelet Transform Modulus Maxima (WTMM) segmentation method (Liu et al., 2021).

Set up your directory structure as follows: <br />
**main_folder <br />
|---- glacier_files <br />
|---- LS8aws <br />**


Workflow:
1) Obtain boxes over each glacier's terminus area as shapefiles. Name them _BoxNameofglacier.shp_ and place in glacier_files. When the BoxNameofglacier folders are generated automatically in preprocess.ipynb, move them into those folders. OPTIONAL: Repeat for each glacier's Randolph Glacier Outline if available. Name them _RGI_BoxNameofglacier.shp_. <br />
2) Run steps in preprocess.ipynb which will downlaod the Landsat 8 image subsets and prep them for WTMM analysis. <br />
3) Run WTMM analysis in WTMM_terminuspick.ipynb. <br />
4) Run steps in analyze_WTMM_results.ipynb to generate the terminus position timeseries along 3 auto-generated flowlines. <br />

