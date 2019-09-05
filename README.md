# automated-glacier-terminus
This repository contains the annotated scripts developed to automatically pick glacier terminus positions in Landsat imagery. This code is developed and managed by Julia "Jukes" Liu (julia.t.liu@gmail.com) under supervision of Dr. Ellyn Enderlin (Boise State University) and Dr. Andre Khalil (University of Maine).

The workflow was developed around detecting glacier termini for 641 of the glaciers around Greenland periphery. The input data required includes boxes (shapefiles) drawn around each glacier's terminus, 2016-2017 glacier ice velocity data (raster), and Landsat path and row over each glacier. These peripheral glaciers are referred to by their terminus BoxID, a three digit code between 001 and 641, referred to as Box###.

## Table of Contents

| Scripts       | Description   |
| ------------- | ------------- |
| LS8_download_aws.ipynb  | Downloads Landsat-8 images available for free through Amazon Web Services (aws)  |
| Gperiph_imgprocessing.ipynb  | Processes Landsat images, Greenland ice velocity, and shapefiles before 2D WTMM analysis  |
| Terminusbox_coords.ipynb  | Pulls vertices of the terminus boxes in pixel coordinates for calculating terminus position  |
| Terminusposition.ipynb  | Calculates and plots terminus positions vs. time and terminus change rates |
| Boxes_topathrows.ipynb  | Determines all the Landsat path_rows for scenes over each glacier |

| Data          | Description   |
| ------------- | ------------- |
| LS_pathrows.csv | Contains Landsat path and row information for each peripheral glacier by their BoxID |
| Box###.shp | Terminus box shapfiles created by Dr. Alison Cook (University of Ottawa) |
| attributes.csv? | Contains buffer distances calculated for each terminus box |
| dir_degree_yx_velocity.tif* | Glacier ice flow direction calculated from x, y velocity data (ESA Cryportal)|
| magnitude_velocity.tif* | Glacier ice flow magnitude (ESA Cryoportal)|

*Files are too large to be uploaded onto GitHub. Contact Julia Liu if you would like to access these data.

## Order of Operations. Workflow.

1) Grab all possible PathRow overlaps for each terminus box (Boxes_topathrows.ipynb)
2) Calculate left side midpoint for each terminus box (Terminusbox_coords.ipynb --> Boxes_coords_pathrows.csv)
2) Create buffer zones around terminus boxes and calculate average glacier flow directions for rotations (Gperiph_preprocess.ipynb)
3) Download subset images and grab dates as a .csv for each image (LS8_download_aws.ipynb --> datetags.csv)
4) Reproject images (Gperiph_preprocess.ipynb)
5) Rotate images (rotations.ijm)
6) Resize images (resize all.ijm)
7) Run 2D WTMM (scr_gaussian.tcl)
8) Pick terminus chain and export to dat files (terminus_pick.tcl --> terminuspicks_metric_yyyy_mm_dd.csv)
9) Calculate centroids for all dat files created (Calculate_term_centroids.ipynb --> trim_centroids_metric.csv)
10) Combine centroids, terminus pick, and datetag csv files to calculate terminus position, change rates, and plot (Terminusposition.ipynb --> terminuschange_Box###_metric.csv)
11) Plot terminus picks (dat files) and centroids over images analyzed, show date and terminus change rate


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

## To-Do:
-separate out individual box shapefiles for every glacier
-determine LS path and row information for every glacier box

