from __future__ import absolute_import

import os
from got10k.datasets import *

from siamfc_pytorch.siamfc.seq_siamfc import SeqTrackerSiamFC
from siamfc_pytorch.siamfc.siamfc import TrackerSiamFC

if __name__ == '__main__':
    # root_dir = os.path.expanduser('../../data/VOT2018')
    root_dir = os.path.expanduser('/srv/s01/leaves-shared/got10k')

    # seqs = GOT10k(root_dir, subset='train', return_meta=True)
    seqs = GOT10k(root_dir, subset='train', return_meta=True)
    tracker = TrackerSiamFC()
    tracker.train_over(seqs)
