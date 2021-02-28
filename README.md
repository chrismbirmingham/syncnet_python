# SyncNet

This repository contains the demo for the audio-to-video synchronisation network (SyncNet). This network can be used for audio-visual synchronisation tasks including: 
1. Removing temporal lags between the audio and visual streams in a video;
2. Determining who is speaking amongst multiple faces in a video. 

Please cite the paper below if you make use of the software. 

## Dependencies
```
pip install -r requirements.txt
```

In addition, `ffmpeg` is required.

Note, the model expects video at 25 fps and audio at 16kHz


## Demo
The demos expect cropped videos from the run_pipeline step below.
SyncNet demo:
```
python demo_syncnet.py --videofile data/example.avi --tmp_dir /path/to/temp/directory
```

Check that this script returns:
```
AV offset:      3 
Min dist:       5.353
Confidence:     10.021
```

## Feature Extraction
This also expects that the videos are cropped with the pipeline.
```
python demo_feature.py --videofile data/example.avi --tmp_dir /path/to/save/features
```


## Pipeline

The pipeline consists of three steps:
1. run_pipeline: extracts video of individual faces into seperate videos. Saves 221x221 video and audio to 
2. run_syncnet: calls the syncnet model on the video streams, gathering features and confidence values
3. run_visualize: combines the detected faces with the confidence in the original video


Full pipeline (these steps are sequential):
```
sh download_model.sh
python run_pipeline.py --videofile /path/to/video.mp4 --reference name_of_video --data_dir /path/to/output
python run_syncnet.py --videofile /path/to/video.mp4 --reference name_of_video --data_dir /path/to/output
python run_visualise.py --videofile /path/to/video.mp4 --reference name_of_video --data_dir /path/to/output
```

Key Outputs:
```
$DATA_DIR/pycrop/$REFERENCE/*.avi - cropped face tracks from run_pipeline
$DATA_DIR/pywork/$REFERENCE/offsets.txt - audio-video offset values from run_syncnet (Not currently written???)
$DATA_DIR/pyavi/$REFERENCE/video_out.avi - output video (as shown below)
```

All Outputs:
```
data_dir/
- pyavi/ref/
  - video.avi (original video in .avi, resampled to 25 FPS)
  - video_only.avi (Video without audio)
  - audio.wav (audio resampled to 16k SR)
  - video_out.avi (output visualization)

- pycrop/ref/
  -000#.avi (224x224 crop around each face-scene detected)
  -000etc...

- pywork/ref/
  - activesd.pckl (distances - a measures of likelihood of talking for each face-frame)
  - faces.pckl (detected faces?)
  - scene.pckl (tracks 'scenes' - continuous detected faces)
  - tracks.pckl (tracks location of detected faces)

- pytmp/ref/
  - every face crop

- pyframes/ref/
  - every frame as a jpg
```

<p align="center">
  <img src="img/ex1.jpg" width="45%"/>
  <img src="img/ex2.jpg" width="45%"/>
</p>

## Publications
 
```
@InProceedings{Chung16a,
  author       = "Chung, J.~S. and Zisserman, A.",
  title        = "Out of time: automated lip sync in the wild",
  booktitle    = "Workshop on Multi-view Lip-reading, ACCV",
  year         = "2016",
}
```
