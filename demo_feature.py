#!/usr/bin/python
# -*- coding: utf-8 -*-

import time
import pdb
import argparse
import subprocess

from SyncNetInstance import *

# ==================== LOAD PARAMS ====================


parser = argparse.ArgumentParser(description="SyncNet")

parser.add_argument(
    "--initial_model", type=str, default="data/syncnet_v2.model", help=""
)
parser.add_argument("--batch_size", type=int, default="20", help="")
parser.add_argument("--vshift", type=int, default="15", help="")  # vshift isn't used?
parser.add_argument("--videofile", type=str, default="data/example.avi", help="")
parser.add_argument("--tmp_dir", type=str, default="data", help="")
parser.add_argument("--save_as", type=str, default="data/features.pt", help="")
parser.add_argument("--feat_type", type=str, default="both", help="both or lip")


opt = parser.parse_args()

setattr(opt, "feat_dir", os.path.join(opt.data_dir, "pyfeat"))

# ==================== RUN EVALUATION ====================

s = SyncNetInstance()

s.loadParameters(opt.initial_model)
print("Model %s loaded." % opt.initial_model)

if opt.feat_type == "lip":
    feats = s.extract_im_feature(opt, videofile=opt.videofile)
    torch.save(feats, opt.save_as)
if opt.feat_type == "both":
    im_feat, cc_feat = s.extract_im_feature(opt, videofile=opt.videofile)
    torch.save(im_feat, os.path.join(opt.feat_dir, opt.reference, "vid_feats.pt"))
    torch.save(cc_feat, os.path.join(opt.feat_dir, opt.reference, "aud_feats.pt"))
