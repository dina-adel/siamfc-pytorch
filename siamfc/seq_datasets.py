from __future__ import absolute_import, division
import cv2
from .datasets import Pair

__all__ = ['Pair']


class SeqPair(Pair):

    def __getitem__(self, index):
        # index = self.indices[index % len(self.indices)]
        if self.frame_index >= len(self.seqs[self.seq_index][0]) - 1:
            self.seq_index = self.seq_index + 1  # if this seq is done, go to next one
            self.frame_index = 0  # reset the frame index to 0

        # get filename lists and annotations
        if self.return_meta:
            img_files, anno, meta = self.seqs[self.seq_index]
            vis_ratios = meta.get('cover', None)
        else:
            img_files, anno = self.seqs[self.seq_index][:2]
            vis_ratios = None

        if self.frame_index == 0:
            # print("Reset hidden is set to True")
            self.reset_hidden = True
        else:
            # print("Reset hidden is set to False")
            self.reset_hidden = False

        # sample a frame pair
        rand_z, rand_x1, rand_x2 = self._sample_pair_with_prev()

        z = cv2.imread(img_files[rand_z], cv2.IMREAD_COLOR)
        x1 = cv2.imread(img_files[rand_x1], cv2.IMREAD_COLOR)
        x2 = cv2.imread(img_files[rand_x2], cv2.IMREAD_COLOR)

        z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
        x1 = cv2.cvtColor(x1, cv2.COLOR_BGR2RGB)
        x2 = cv2.cvtColor(x2, cv2.COLOR_BGR2RGB)

        box_z = anno[rand_z]
        box_x1 = anno[rand_x1]

        item = (z, [x1, x2], box_z, box_x1)
        if self.transforms is not None:
            item = self.transforms(*item)

        return item, self.reset_hidden

    def _sample_pair_with_prev(self):
        rand_z = self.frame_index
        rand_x2 = rand_z
        rand_x1 = rand_z + 1
        rand_x1 = rand_x1 if rand_x1 <= len(self.seqs[self.seq_index][0]) - 1 else rand_z
        self.frame_index = self.frame_index + 1  # increment frame index
        return rand_z, rand_x1, rand_x2
