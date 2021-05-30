'''Dataset used to train the front-back model'''

import torch.utils.data as D
import torch
from torchvision import transforms
from typing import Union, Sequence
from PIL import Image

from .. import root_dir


class FrontBackDataset(D.Dataset):
    '''Dataset used to train the front-back model'''
    def __init__(self):
        # Define the directory containing all the images
        motion_dir = root_dir / 'data' / 'motion'

        # Get the lists of file names related to the three classes
        enter_fnames = list((motion_dir / 'enter').glob('*.jpg'))
        leave_fnames = list((motion_dir / 'leave').glob('*.jpg'))
        other_fnames = list((motion_dir / 'other').glob('*.jpg'))

        # Define the list containing all the file names
        self.all_fnames = []
        self.all_fnames.extend(enter_fnames)
        self.all_fnames.extend(leave_fnames)
        self.all_fnames.extend(other_fnames)

        # Define the indices of the three classes
        enter_idxs = list(range(len(enter_fnames)))
        leave_idxs = [len(enter_fnames) + n for n in range(len(leave_fnames))]
        other_idxs = [len(enter_fnames) + len(leave_fnames) + n
                      for n in range(len(other_fnames))]

        # Define the label tensor
        num_fnames = len(enter_fnames) + len(leave_fnames) + len(other_fnames)
        self.labels = torch.LongTensor(num_fnames)
        self.labels[enter_idxs] = 0
        self.labels[leave_idxs] = 1
        self.labels[other_idxs] = 2

        # Preprocessing function of the images
        self.preprocess = transforms.ToTensor()

    def __getitem__(self, idx: Union[int, Sequence[int]]):
        # Ensure that `idx` is a sequence
        try:
            is_int = False
            idx[0]
        except (IndexError, TypeError):
            is_int = True
            idx = [idx]

        # Get the images with the given indices
        fnames = [self.all_fnames[i] for i in idx]
        tensor = torch.stack([self.preprocess(Image.open(fname).convert('RGB'))
                              for fname in fnames])

        # Get the labels with the given indices
        labels = self.labels[idx]

        # If input was a single integer, then output the tensor and label
        # corresponding to that single instance
        if is_int:
            tensor = tensor[0]
            labels = labels[0]

        # Return the image tensor and the labels
        return tensor, labels

    def __len__(self):
        return len(self.all_fnames)
