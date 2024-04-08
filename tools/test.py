from __future__ import absolute_import

import os
from got10k.experiments import *

from siamfc_pytorch.siamfc import TrackerSiamFC
from siamfc_pytorch.siamfc.seq_siamfc import SeqTrackerSiamFC


if __name__ == '__main__':
    net_path = 'pretrained/recurrentVanilla/siamfc_alexnet_e50.pth'
    tracker = TrackerSiamFC(net_path=net_path)

    # root_dir = os.path.expanduser('../../data/VOT2018')
    # e = ExperimentVOT(root_dir, version=2018, report_dir='reports/recurrentVanilla',
    #                  result_dir='results/recurrentVanilla')
    root_dir = os.path.expanduser('/srv/s01/leaves-shared/got10k')
    e = ExperimentGOT10k(root_dir, subset='test', report_dir='reports/recurrentVanilla', result_dir='results/recurrentVanilla')
    e.run(tracker)
    e.report([tracker.name])
