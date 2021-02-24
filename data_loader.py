from __future__ import print_function, division
import os
import subprocess
import torch
import pandas as pd
import glob
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from shutil import rmtree
from scipy.io import wavfile
import numpy as np
import cv2
import python_speech_features
import math
from tqdm import tqdm


# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class SingleVideoDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, videofile, tmp_dir, reference, batch_size, convert_again=True):
        """
        """
        self.videofile = videofile
        self.tmp_dir = tmp_dir
        self.reference = reference
        self.batch_size = batch_size

        if convert_again:
            self.convert_files(videofile)
        print("loading audio")
        self.load_audio()
        print("loading video")
        self.load_video()
        print("loading completed")
        if (float(len(self.audio)) / 16000) != (float(len(self.flist)) / 25):
            print(
                "WARNING: Audio (%.4fs) and video (%.4fs) lengths are different."
                % (float(len(self.audio)) / 16000, float(len(self.flist)) / 25)
            )

        self.min_length = min(len(self.flist), math.floor(len(self.audio) / 640))
        self.lastframe = self.min_length - 5

    def convert_files(self, videofile):
        if os.path.exists(os.path.join(self.tmp_dir, self.reference)):
            rmtree(os.path.join(self.tmp_dir, self.reference))

        os.makedirs(os.path.join(self.tmp_dir, self.reference))

        command = "ffmpeg -y -i %s -threads 1 -f image2 %s" % (
            videofile,
            os.path.join(self.tmp_dir, self.reference, "%06d.jpg"),
        )
        _ = subprocess.call(command, shell=True, stdout=None)

        command = (
            "ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s"
            % (videofile, os.path.join(self.tmp_dir, self.reference, "audio.wav"))
        )
        _ = subprocess.call(command, shell=True, stdout=None)

    def load_video(self):
        self.images = []

        self.flist = glob.glob(os.path.join(self.tmp_dir, self.reference, "*.jpg"))
        self.flist.sort()
        print(f"Found {len(self.flist)} frames")

    def load_audio(self):
        sample_rate, self.audio = wavfile.read(
            os.path.join(self.tmp_dir, self.reference, "audio.wav")
        )
        mfcc = zip(*python_speech_features.mfcc(self.audio, sample_rate))
        mfcc = np.stack([np.array(i) for i in mfcc])

        self.cc = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0)
        self.cct = torch.autograd.Variable(
            torch.from_numpy(self.cc.astype(float)).float()
        )
        print(f"audio is shaped: {self.cct.shape}")

    def __len__(self):
        # return int(self.lastframe / self.batch_size) # for #old__getitem__
        return int(self.lastframe)

    def __getitem__(self, idx):
        self.images = []

        for fname in self.flist[idx : idx + 5]:
            self.images.append(cv2.imread(fname))
        if len(self.images) < 5:
            print(
                f"Asked for {idx} which is {i} out of {len(self.flist)}. [second index {i + self.batch_size + 5}"
            )

        im = np.stack(self.images, axis=3)
        im = np.expand_dims(im, axis=0)
        im = np.transpose(im, (0, 3, 4, 1, 2))

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())
        cctv = self.cct[:, :, :, idx * 4 : idx * 4 + 20]

        return imtv, cctv

    def old__getitem__(self, idx):

        i = idx * self.batch_size
        self.images = []

        for fname in self.flist[i : i + self.batch_size + 5]:
            self.images.append(cv2.imread(fname))
        if len(self.images) == 0:
            print(
                f"Asked for {idx} which is {i} out of {len(self.flist)}. [second index {i + self.batch_size + 5}"
            )

        im = np.stack(self.images, axis=3)
        im = np.expand_dims(im, axis=0)
        im = np.transpose(im, (0, 3, 4, 1, 2))

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())

        im_batch = [
            imtv[:, :, vframe : vframe + 5, :, :]
            for vframe in range(0, min(self.lastframe - i, self.batch_size))
        ]
        if len(im_batch) == 0:
            print(
                f"2-Asked for {idx} which is {i} out of {len(self.flist)}. [second index {i + self.batch_size + 5}"
            )
            print(len(self.images))
            print(imtv.shape)
            print(self.lastframe - i, self.batch_size)
            print("no tenser here!!!!!!!!!!!!!!!!")
        im_in = torch.cat(im_batch, 0)

        cc_batch = [
            self.cct[:, :, :, vframe * 4 : vframe * 4 + 20]
            for vframe in range(i, min(self.lastframe, i + self.batch_size))
        ]
        cc_in = torch.cat(cc_batch, 0)

        return im_in, cc_in


if __name__ == "__main__":
    videofile = "/media/chris/M2/1-Raw_Data/Videos/1/cropped/reseampled_center.mp4"
    tmp_dir = "/media/chris/M2/1-Raw_Data/syncnet_output/pytmp"
    reference = "DL_test"
    batch_size = 50
    svd = SingleVideoDataset(
        videofile, tmp_dir, reference, batch_size, convert_again=False
    )
    # print(len(svd))
    # for i in range(len(svd)):
    #     print(i)
    #     v, a = svd[i]
    #     if i > 10:
    #         break

    dataloader = DataLoader(svd, batch_size=50, shuffle=False, num_workers=5)
    i = 0
    for v, a in tqdm(dataloader):
        v = torch.squeeze(v, dim=1)
        a = torch.squeeze(a, dim=1)
        print(i, v.shape, a.shape)
        i += 1
        if i > 100:
            break

