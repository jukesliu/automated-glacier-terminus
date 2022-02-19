# automated-glacier-terminus

This repository contains code to automatically delineate glacier terminus positions in Landsat 8 imagery using the adapted 2D Wavelet Transform Modulus Maxima (WTMM) segmentation method [(Liu et al., 2021)](https://ieeexplore.ieee.org/document/9349100 "doi: 10.1109/TGRS.2021.3053235"). There are two workflows, one exclusively run in Python and the other which requires the Xsmurf software. Please contact jukesliu@u.boisestate.edu and andre.khalil@maine.edu if you would like to install Xsmurf.

__Set up a local conda environment__ <br />
From the repository root, run in a terminal:
- conda env create -f environment.yml

activate newly created environment:

- conda activate autoterm_env

then start jupyter in browser

- jupyter notebook &
    
__Set up your directory structure as follows:__ <br />
main_folder <br />
|---- glacier_files <br />
|---- LS8aws <br />


__Python-only workflow:__
1) Obtain boxes over each glacier's terminus area as shapefiles. Name them _BoxNameofglacier.shp_ and place in glacier_files. When the BoxNameofglacier folders are generated automatically in __preprocess.ipynb__, move them into their respective folders. OPTIONAL: Repeat for each glacier's Randolph Glacier Inventory outline if available. Name them _RGI_BoxNameofglacier.shp_. <br />
2) Run steps in __preprocess.ipynb__ which will download the Landsat 8 image subsets and prep them for WTMM analysis. <br />
3) Run the automated terminus delineation with __wtmm2d_terminuspick.ipynb__. <br />
4) Run steps in __analyze_wtmm_results.ipynb__ to generate 3 flowlines within the glacier terminus boxes and produce the time series of terminus position along each of the three flowlines. <br />

__Workflow with Xsmurf:__
1) Obtain boxes over each glacier's terminus area as shapefiles. Name them _BoxNameofglacier.shp_ and place in glacier_files. When the BoxNameofglacier folders are generated automatically in __preprocess.ipynb__, move them into their respective folders. OPTIONAL: Repeat for each glacier's Randolph Glacier Inventory outline if available. Name them _RGI_BoxNameofglacier.shp_. <br />
2) Run steps in __preprocess.ipynb__ which will download the Landsat 8 image subsets and prep them for WTMM analysis. <br />
3) Run the automated terminus delineation with __wtmm2d_terminuspick_Xsmurf.ipynb__. <br />
4) Run steps in __analyze_wtmm_results.ipynb__ to generate 3 flowlines within the glacier terminus boxes. For step 3, uncomment and run the section under "Xsmurf workflow" to produce the time series of terminus position along each of the three flowlines.<br />
