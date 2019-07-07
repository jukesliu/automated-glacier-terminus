# automated-glacier-terminus
This repository contains the annotated scripts developed to automatically pick glacier terminus positions in Landsat imagery. This code is developed and managed by Julia Liu (julia.t.liu@gmail.com) under supervision of Dr. Ellyn Enderlin (Boise State University) and Dr. Andre Khalil (University of Maine).

The workflow was developed around detecting glacier termini for 641 of the glaciers around Greenland periphery. The input data required includes boxes (shapefiles) drawn around each glacier's terminus, 2016-2017 glacier ice velocity data (raster.TIF), and Landsat path and row over each glacier. These peripheral glaciers are referred to by their terminus BoxID, a three digit code between 001 and 641, referred to as Box###.

## Table of Contents

| Scripts       | Description   |
| ------------- | ------------- |
| LS8_download_aws.ipynb  | Downloads Landsat-8 images available for free through Amazon Web Services (aws)  |
| Gperiph_imgprocessing.ipynb  | Processes Landsat images, Greenland ice velocity, and shapefiles before 2D WTMM analysis  |

| Data          | Description   |
| ------------- | ------------- |
| LS_pathrows.csv | Contains Landsat path and row information for each peripheral glacier by their BoxID |
| Box###.shp | Terminus box shapfiles created by Dr. Alison Cook (University of Ottawa) |
| attributes.csv? | Contains buffer distances calculated for each terminus box |
| dir_degree_yx_velocity.tif | Glacier ice flow direction calculated from x, y velocity data (ESA Cryportal) |
| magnitude_velocity.tif | Glacier ice flow magnitude (ESA Cryoportal) |

## File organization
The workflow follows a certain file structure for storing the input and output data for each glacier. The general structure is:

