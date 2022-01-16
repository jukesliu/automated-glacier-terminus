#!/bin/bash
#####################################################################
# By Jukes Liu

# Run this bash script from the terminal on the glacier terminus detection
# linux machine
#
# Last updated 2020 05 24
######################################################################

# set necessary parameters to constant for the following functions
order='_MSA'
V=3
N1=1
N2=1

## grab input csv files
#download_csv=$1
#date_csv=$2
#centerline_csv=$3
#vel_csv=$4

# set list of glaciers and list of CPUs
num_inputs=$#
# read the BoxIDs input into a string array and split
inputs=$@
read -a strarr <<< $inputs
IDs=${strarr[@]}
read -a BoxIDs <<< $IDs
declare -i num_glaciers=${#BoxIDs[@]}
echo ${BoxIDs[@]}
echo "Number of glaciers: $num_glaciers"

CPUs=('0x1' '0x2' '0x4' '0x8' '0x10' '0x20' '0x40' '0x80')
num_CPUs=8

declare -i num_glaciers=${#BoxIDs[@]}

# Define the paths
path=/media/jukes/jukes1/LS8aws/
outpath=/home/jukes/Documents/Sample_glaciers/
path_script=/home/jukes/Documents/Scripts/terminus_pick${order}_1glacier_nd.tcl
path_xsn=/home/akhalil/src/xsmurf-2.7/main/xsmurf

# set threshold combination
for size_thresh in 0.4;
do
    for mod_thresh in 0.7;
    do
	for arg_thresh in 0.1;
	do
	    echo "Size threshold: $size_thresh"
	    echo "Mod threshold: $mod_thresh"
	    echo "Arg threshold: $arg_thresh"

	    # Pick the terminus via parallel computing using the threshold combination
	    # Calculate number of batches
	    num_batches=$(expr ${num_glaciers} / ${num_CPUs})
	    # grab number in the last batch (remainder)
	    num_lastbatch=$(expr ${num_glaciers} % ${num_CPUs})
	    # deal with rounding:
	    if (($num_lastbatch > 0))
	    then
		num_batches=$(expr $num_batches + 1)
	    else
		num_lastbatch=8
	    fi
	    
	    echo "${num_batches} batches"
	    echo "$num_lastbatch in last batch"
	    
	    counter=0
	    
	    # Parallel terminus picking: 
	    for ((batch=1; batch < ($num_batches+1); batch++ )); do
		if (( $batch < $num_batches )); then
		    echo "Start batch $batch out of $num_batches"
		    # store PIDs
		    PIDs=()
		    for ((CPU=0; CPU < ($num_CPUs); CPU++)); do
			# grab the BoxID
			BoxID=${BoxIDs[$counter]}
			# run the script for that glacier
			taskset ${CPUs[$CPU]} $path_xsn -nodisplay $path_script $size_thresh $mod_thresh $arg_thresh $BoxID &
			# grab the PID
			PID=$!
			PIDs+=($PID)
			counter=$counter+1
		    done

		    #wait for these to finish before continuing
		    wait ${PIDs[@]}
		# for the last batch:
		else
		    echo "Start batch $batch out of $num_batches"
		    PIDs=()
		    for ((CPU=0; CPU < ($num_lastbatch); CPU++)); do
			# grab the BoxID
			BoxID=${BoxIDs[$counter]}
			# run the script for that glacier
			taskset ${CPUs[$CPU]} $path_xsn -nodisplay $path_script $size_thresh $mod_thresh $arg_thresh $BoxID &
			# grab the PID
			PID=$!
			PIDs+=($PID)
			counter=$counter+1
		    done

		    #wait for these to finish before continuing
		    wait ${PIDs[@]}
		fi
	    done
	done
    done
done
