from __future__ import absolute_import

import os
import glob
import numpy as np

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    seq_dir = os.path.expanduser('data/VOT-2018/crossing/')
    img_files = sorted(glob.glob(seq_dir + '*.jpg'))
    anno = np.loadtxt(seq_dir + 'groundtruth.txt', delimiter=',')

    x_tl = anno[0][0]  # Top-left x-coordinate
    y_tl = anno[0][1]  # Top-left y-coordinate
    width = anno[0][4] - anno[0][0]  # Width calculation: bottom-right x - top-left x
    height = anno[0][5] - anno[0][1]  # Height calculation: bottom-right y - top-left y

    # Create a list of 4 values: [x, y, width, height]
    bbox_4_values = [x_tl, y_tl, width, height]

    net_path = 'pretrained/siamfc_alexnet_e50.pth'
    tracker = TrackerSiamFC(net_path=net_path)
    tracker.track(img_files, bbox_4_values, visualize=True)
