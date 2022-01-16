#!/bin/bash
#####################################################################
# By Jukes Liu

# Run this bash script from the terminal on the glacier terminus detection
# linux machine via:
# cd /home/jukes/Documents/Scripts/; chmod +x WTMM_parallel_test.sh; ./WTMM_parallel_test.sh 120 174
#
# Last updated 2020 07 01
######################################################################
round() {
    printf "%.{$2:-0}f" "$1"
}

# catch BoxIDs input by the user
inputs=$@
read -a strarr <<< $inputs
IDs=${strarr[@]}
read -a BoxIDs <<< $IDs

# Set list of CPUs
CPUs=('0x1' '0x2' '0x4' '0x8' '0x10' '0x20' '0x40' '0x80')
num_CPUs=${#CPUs[@]}

# Define the paths
path=/media/jukes/jukes1/LS8aws/
path_script=/home/jukes/Documents/Scripts/scr_gaussian_image.tcl
path_xsn=/home/akhalil/src/xsmurf-2.7/main/xsmurf

# for box in BoxID
for BoxID in ${BoxIDs[@]};
do
    echo "WTMM for Box$BoxID"
    imagepath=${path}Box${BoxID}/rotated_c1/

    # grab images and number of images
    num_images=0
    image_list=()
    IDlength=$(expr length $BoxID)
    for entry in ${imagepath}*${BoxID}_PS.pgm
    do
	num_images=$(expr $num_images + 1)
	begin=$(expr 45+$IDlength-3)
	end=$(expr 62+$IDlength-3)
	imagename=${entry:$begin:$end}
	image_list+=($imagename)
    done

    echo "$num_images images"
    # echo "${image_list[@]}"
    
    # calculate number of batches
    num_batches=$(expr ${num_images} / ${num_CPUs})
    # grab number in the last batch (remainder)
    num_lastbatch=$(expr ${num_images} % ${num_CPUs})
    # deal with rounding:
    if (($num_lastbatch > 0))
    then
	num_batches=$(expr $num_batches + 1)
    else
	num_lastbatch=8
    fi
    
    echo "${num_batches} batches"
    echo "$num_lastbatch in last batch"

    # parallel WTMM:
    counter=0
    # echo ${image_list[@]}
    for ((batch=1; batch < ($num_batches+1); batch++ )); do	
	if (( $batch < $num_batches ))
	then
	    echo "Start batch $batch out of $num_batches"
	    # store PIDs
	    PIDs=()
	    for ((CPU=0; CPU < ($num_CPUs); CPU++)); do
		# grab the image
		image=${image_list[$counter]}
		echo "$image"
		# run the script for that glacier
		taskset ${CPUs[$CPU]} $path_xsn -nodisplay $path_script $BoxID $image &
		# grab the PID
		PID=$!
		PIDs+=($PID)
		counter=$(expr $counter + 1)
	    done

	    #wait for these to finish before continuing
	    wait ${PIDs[@]}
	    # for the last batch:
	else
	    echo "Start batch $batch out of $num_batches"
	    PIDs=()
	    for ((CPU=0; CPU < ($num_lastbatch); CPU++)); do
		# grab the image
		image=${image_list[$counter]}
		echo "$image"
		# run the script for that glacier
		taskset ${CPUs[$CPU]} $path_xsn -nodisplay $path_script $BoxID $image &
		# grab the PID
		PID=$!
		PIDs+=($PID)
		counter=$(expr $counter + 1)
	    done

	    #wait for these to finish before continuing
	    wait ${PIDs[@]}
	fi
    done
done
