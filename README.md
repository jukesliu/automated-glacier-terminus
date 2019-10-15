# automated-glacier-terminus
This repository contains the annotated scripts developed to automatically pick glacier terminus positions in Landsat imagery. This code is developed and managed by Julia "Jukes" Liu (julia.t.liu@gmail.com) under supervision of Dr. Ellyn Enderlin (Boise State University) and Dr. Andre Khalil (University of Maine).

The workflow was developed around detecting glacier termini for 641 of the glaciers around Greenland periphery. The input data required includes boxes (shapefiles) drawn around each glacier's terminus, 2016-2017 glacier ice velocity data (raster), and Landsat path and row over each glacier. These peripheral glaciers are referred to by their terminus BoxID, a three digit code between 001 and 641, referred to as Box###.

## Table of Contents

| Scripts       | Description   |
| ------------- | ------------- |
| AWS_LS8_download.ipynb  | Downloads Landsat-8 images available over the glaciers through Amazon Web Services (aws)  |
| Gperiph_imgprocessing.ipynb  | Processes Landsat images, Greenland ice velocity, and shapefiles |
| Terminusbox_coords.ipynb  | Pulls vertices of the terminus boxes in pixel coordinates for calculating terminus position  |
| Terminusposition.ipynb  | Calculates and plots terminus positions vs. time and terminus change rates |
| Show_term_pick_results.ipynb  | Visualizes the terminus pick results over the images analyzed |

| Data          | Description   |
| ------------- | ------------- |
| LS_pathrows.csv | Contains Landsat path and row information for each peripheral glacier by their BoxID |
| Box###.shp | Terminus box shapfiles created by Dr. Alison Cook (University of Ottawa) |
| attributes.csv? | Contains buffer distances calculated for each terminus box |
| dir_degree_yx_velocity.tif* | Glacier ice flow direction calculated from x, y velocity data (ESA Cryportal)|
| magnitude_velocity.tif* | Glacier ice flow magnitude (ESA Cryoportal)|

*Files are too large to be uploaded onto GitHub. Contact Julia Liu if you would like to access these data.

## Order of Operations. Workflow.

1) Create buffer zones around terminus boxes, rasterize terminus boxes, and calculate average glacier flow directions for rotations (Gperiph_preprocess.ipynb --> Glacier_velocities.csv)
2) Download subset images for all LS Path_Row combos over the glacier, reproject, and grab dates as a .csv for each image (AWS_LS8_download.ipynb --> imgdates.csv)
3) Convert downloaded subsets to png files (mogrify -format png *.TIF) 
4) Rotate terminus box rasters so flow direction is due right (add into script)
5) Rotate all images so flow direction is due right (rotations.ijm)
6) Resize all images so that mask and img dimensions are uniform (resize_all.ijm)

7) Run 2D WTMM (scr_gaussian.tcl)
8) Pick top 5 terminus chains and export to dat files, record info in a .csv file (terminus_pick.tcl --> terminuspicks_metric_yyyy_mm_dd.csv)

9) Calculate left side midpoint and centerlines for each rotated terminus box (Terminusbox_coords.ipynb --> Boxes_coords_pathrows.csv)
10) Calculate centerline intersection with each terminus picks, combine with imgdates.csv and WTMM info to plot terminus position timeseries and terminus change rates (Terminusposition.ipynb)
11) Plot terminus picks (dat files) and centroids over images analyzed (Show_term_pick_results.ipynb)


## File organization
The workflow follows a certain file structure for storing the input and output data for each glacier. The general structure is:
Scene_Directory
> __Glacier folders by BoxID (Box001)__
> Box002
> Box###
> ...
> __Path_Row folders containing scene metadata__
> Path###_Row###
> ...
> __terminus pick results by metric__
> terminus_highestmass
  > Box### 
  > Box###
> terminus_highestsize
