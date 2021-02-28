#!/bin/bash 
VIDEO="/media/chris/M2/1-Raw_Data/Videos/1/cropped/left.mp4"
REF="1l"
DATA="/media/chris/M2/2-Processed_Data/syncnet_output"
#echo "Start Pipeline"
#python run_pipeline.py --videofile $VIDEO --reference $REF --data_dir $DATA
echo "Start syncnet"
python run_syncnet.py --videofile $VIDEO --reference $REF --data_dir $DATA
echo "Start visualize"
python run_visualise.py --videofile $VIDEO --reference $REF --data_dir $DATA