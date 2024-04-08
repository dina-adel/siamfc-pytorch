from __future__ import absolute_import, division, print_function
from __future__ import absolute_import, division, print_function

import sys
import os
from torch.utils.data import DataLoader
from .transforms import SiamFCTransforms

import torch
from .backbones import *
from .siamfc import TrackerSiamFC, Net
from .heads import SiamFC
from .seq_datasets import SeqPair

__all__ = ['SeqTrackerSiamFC']


class SeqTrackerSiamFC(TrackerSiamFC):

    def __init__(self, net_path=None, **kwargs):
        super(SeqTrackerSiamFC, self).__init__()
        self.net = Net(
            backbone=SeqFramesAttenAlexNet(),
            head=SiamFC(self.cfg.out_scale))
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

    def train_step(self, data, backward=True):
        # set network mode
        self.net.train(backward)

        batch = data[0]
        reset_hidden = data[1]

        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x1 = batch[1][0].to(self.device, non_blocking=self.cuda)
        x2 = batch[1][1].to(self.device, non_blocking=self.cuda)

        with torch.set_grad_enabled(backward):
            # inference
            responses = self.net(z, [x1, x2], reset_hidden)

            # calculate loss
            labels = self._create_labels(responses.size())
            loss = self.criterion(responses, labels)

            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss.item()

    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None,
                   save_dir='pretrained/seqFramesAtten'):
        self.net.train()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)

        dataset = SeqPair(
            seqs=seqs,
            transforms=transforms)

        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,  # we are using rnn, so better not shuffle the dataset
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)

        # loop over epochs
        for epoch in range(self.cfg.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)

            # loop over dataloader
            for it, data in enumerate(dataloader):
                loss = self.train_step(data, backward=True)
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                sys.stdout.flush()

            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)
