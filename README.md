# automated-glacier-terminus
Code to automatically pick glacier terminus positions in Landsat imagery.

Workflow:
1) Create box raster and buffer zone around glacier terminus box (Gperiph_preprocess.ipynb)
2) Calculate flow directions and max speeds, etc. (Gperiph_preprocess.ipynb)
3) Download images using the boxes and buffers and reproject (AWS_LS8_download.ipynb)
4) Convert all reprojected images from .TIFs to .pngs (mogrify -format png *.TIF)
5) Rotate the images and box raster using flow direction so flow is due right (rotate_LS.ijm)
6) Resize all images so they have the exact same dimensions (resize_all.ijm)
7) Run the 2D WTMM (scr_gaussian.tcl)
8) Calculate terminus box midpoint and centerline (Terminusbox_coords.ipynb)
9) Grab centerline pixel ratios (Pixel_brightness_centerline.ipynb)
10) Pick the top terminus chains from the extrema images from step 7 (terminus_pick.tcl)
11) Visualize the results (Results_full_run.ipynb OR Results_per_glacier.ipynb)
12) Calculate terminus position and plot timeseries (Results_per_glacier.ipynb)
13) Plot multiple time series as subplots (Timeseries_plots.ipynb)
