# automated-glacier-terminus

This repository contains python code to automatically delineate glacier terminus positions in Landsat 7 and Landsat 8 imagery using the adapted 2D Wavelet Transform Modulus Maxima (WTMM) segmentation method [(Liu et al., 2021)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9349100). The 2D WTMM method calculates gradients in brightness throughout each image at 50 unique spatial scales and produces maxima chains correpsonding to the maximum brightness gradients. These maxima chains delineate regions of high contrast in brightness, such as the glacier terminus margin. Relative terminus positions are calculated along three flowlines across the terminus width, as such:

    

For more details, please see:

    Liu,J., Enderlin, E. M., Marshall, H.P., and A. Khalil (2021). Automated Detection of Marine Glacier Calving Fronts Using the 2-D Wavelet Transform Modulus Maxima Segmentation Method. IEEE Transactions on Geoscience and Remote Sensing, vol. 59, no. 11, pp. 9047-9056. doi: 10.1109/TGRS.2021.3053235

We politely request that you provide attribution to this work if you intend to use the code for published research by citing the paper above. The repo itself is citable as:

    Citation 

There are two workflows, one exclusively run in Python and the other which requires the Xsmurf software. Please contact jukesliu@u.boisestate.edu and andre.khalil@maine.edu if you would like to install Xsmurf.


## Set up your working directory structure as follows: <br />
* glacier_files/
    + Box001/
    + Box002/
    + BoxNNN/
* LSaws/
    + Box001/
    + Box002/
    + BoxNNN/
             
Begin by obtaining or generating rectangular boxes over each glacier's terminus area as ESRI shapefiles. Name them _BoxNAMEID.shp_. These will be stored in your **glacier_files/** folder (basepath variable). When the BoxID subfolders are generated automatically in the first two scripts, move the shapefiles into their respective folders. OPTIONAL: Repeat for each glacier's Randolph Glacier Inventory outline if available. Name them _RGI_BoxNAMEID.shp_. <br />

The **LSaws** folder (downloadpath variable) will contain the Landsat images automatically downloaded in the first script. <br />


## Run the scripts in this order: <br />
1) LS_image_download_AWS.ipynb*  <br />
2) preprocess.ipynb  <br />
3) wtmm2d_terminuspick.ipynb (python only) OR wtmm2d_terminuspick_Xsmurf.ipynb (requires Xsmurf)  <br />
4) analyze_wtmm_results.ipynb  <br />

_*The first script in this workflow automatically downloads subsets of Landsat 7 and 8 imagery available at low cost through the Amazon Web Services (AWS) s3 bucket. This requires a requester pays account through AWS. If you would like to input your own set of Landsat 7 and 8 images, skip the first script and start with preprocess.ipynb._  <br />

Each .ipynb file contains text describing the analysis in that step.  <br />

## System requirements: <br />
Requires GDAL version 3.2 or newer with command terminal functionality (e.g., gdalwarp, gdal_translate). Several steps require the [ImageMagick command line software](https://imagemagick.org/script/download.php) to be installed. If downloading images through AWS, the AWS command line interface must also be installed.

Other packages required are listed in the environment.yml file included. **Set up a local conda environment** with the .yml file. From the repository root, run in a terminal:

- conda env create -f environment.yml

activate newly created environment:

- conda activate autoterm_env

then start jupyter in browser

- jupyter notebook &
