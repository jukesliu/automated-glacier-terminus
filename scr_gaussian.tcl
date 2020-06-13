# prints process ID number
puts [pid] 

set theScr {
    
    # initialize all parameters for the WTMM study
    # change path to the parameters_gaussian.tcl path
    init -force -filename /home/jukes/Documents/Scripts/parameters_gaussian.tcl
    
    set logCmd dputst

    #set list of BoxIDs
    set BoxIDs $argv
    
    foreach BoxID $BoxIDs {
	puts $BoxID
	
	# change path to your rotated images
	set path ${baseDir}/LS8aws/Box${BoxID}/rotated_c1/
	puts "Path $path"
	
	# set a list of images to be analyzed
	set image_list [glob -dir $path "*${BoxID}_PS.pgm"]
	#puts $image_list
    
	foreach imaIdf $image_list {	
	    dputs " wtmm $wavelet"
	
	    #create folder with processed images
	    if (![file exists ${imaIdf}_max_${wavelet}]) {
		file mkdir ${imaIdf}_max_${wavelet}

		puts "folder created"
	    
	    }
	    
	    #load 16bit image in
	    iload ${imaIdf} c -pgm16bit -swap_bytes
        
	    # loop over scalar fields
	    wtmmg c {
		esave max$scaleIdF ${imaIdf}_max_${wavelet}/max$scaleIdF
		delete mod$scaleIdF
		delete arg$scaleIdF
	    
	    }
	    dputs "End wtmm."
	
	}
    }
}
ist $theScr

#exit
