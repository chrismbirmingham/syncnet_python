#!/bin/bash 
# A simple example of the full pipeline for many videos

for study_idx in {1..27}; do
    VIDEO="/media/chris/M2/1-Raw_Data/Videos/${study_idx}/cropped/center.mp4"
    REF="${study_idx}c"
    DATA="/media/chris/M2/2-Processed_Data/syncnet_output"
    echo "Start Pipeline"
    python run_pipeline.py --videofile $VIDEO --reference $REF --data_dir $DATA
    echo "Start syncnet"
    python run_syncnet.py --videofile $VIDEO --reference $REF --data_dir $DATA --batch_size 50
    echo "Start visualize center"
    python run_visualise.py --videofile $VIDEO --reference $REF --data_dir $DATA
  
done

