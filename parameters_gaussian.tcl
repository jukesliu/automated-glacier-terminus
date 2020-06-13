studyName {test_wtmm2d_scalarfield}

#Check the base directory!
# set baseDir /home/jukes/Documents/Sample_glaciers
set baseDir /media/jukes/jukes1

wavelet gaussian

#can switch to GPU for faster processing
set useFftw 1
set useNMaxSup 1
set useGPU 0
set path2cuda /home/pkestene/install/nvidia/cuda/NVIDIA_CUDA_SDK1.1/bin/linux/release
set nThreads 1

#size of image, minimum scale = amin*7pixels, noct = number of octaves, nvox = number of voices (scales within an octave)
set size 1500
set amin 1
set noct 5
set nvox 10

set nbox_mod 1500
set nbox_arg 1500
set nbox_grax 20000
set nbox_gray 20000
set calendos_size 512

#use border percent to set a buffer around window
set border_percent 0.72

#change the wavelet from gaussian to mexican
set similitude 0.8
set isgaussian 1
set ismexican 0
set issave 1
set angle 0
set ishisto 1
set ismaxhisto 1
set ishistoinit 0
set iscontpart 0
set isthetapart 0

#Multifractal analysis, roughness 
#set q_lst {-4 -3.5 -3 -2.5 -2 -1.5 -1 -0.5 -0.3 -0.2 -0.1 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.5 1.6 1.8 2 2.2 2.4 2.5 2.6 2.8 3 3.2 3.4 3.5 3.6 3.8 4 5}

set q_lst {-6 -5 -4 -3 -2 -1.5 -1 -0.8 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 2.5 3 3.5 4 5 6 7 8}

set H 0.5

set theta_lst {0 0.785 m0 0.7854 0.785 mpi4 1.5708 0.785 mpi2 2.3562 0.785 m3pi4 3.14159 0.785 m0 -0.7854 0.785 mpi4 -1.5708 0.785 mpi2 -2.3562 0.785 m3pi4}
set dtheta_histo 0.3925
set theta_lsthisto {0 0.7854 1.5708 2.3562  3.14159 -0.7854 -1.5708 -2.3562}


