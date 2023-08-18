import os
import glob
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class ACDC(Dataset):
    def __init__(
        self, root, train=True, transform=None, client_id=None, out_client=False
    ):
        self.root = root
        self.split = "train" if train else "val"

        self.transform = transform
        self.client_id = client_id
        self.out_client = out_client
        self.ignore_index = 255
        self.images, self.targets = [], []
        self.domains = []

        for fname in glob.iglob(self.root + "/rgb/**/*.png", recursive=True):
            if f"/{self.split}/" in fname:
                self.images.append(fname)
                fname = fname.replace("rgb", "gt")
                fname = fname.replace("anon", "labelTrainIds")
                self.targets.append(fname)

                if "/rain/" in fname:
                    self.domains.append(0)
                elif "/fog/" in fname:
                    self.domains.append(1)
                elif "/snow/" in fname:
                    self.domains.append(2)
                elif "/night/" in fname:
                    self.domains.append(3)

        self.domains = np.array(self.domains)
        self._build_truncated_dataset()

    def _build_truncated_dataset(self):

        ###### Temporaral code for mini-sized experiment ############
        use_dataidxs = []

        for client_id in range(4):
            dataidxs = np.where(self.domains == client_id)[0].tolist()
            use_dataidxs += dataidxs[:100]

        self.images = [self.images[idx] for idx in use_dataidxs]
        self.targets = [self.targets[idx] for idx in use_dataidxs]
        self.domains = self.domains[use_dataidxs]
        #############################################################

        if self.client_id is not None:
            if self.out_client:
                dataidxs = np.where(self.domains != self.client_id)[0]
            else:
                dataidxs = np.where(self.domains == self.client_id)[0]
            self.images = [self.images[idx] for idx in dataidxs]
            self.targets = [self.targets[idx] for idx in dataidxs]
            self.domains = self.domains[dataidxs]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.array(Image.open(self.images[index]).convert("RGB"))
        target = np.array(Image.open(self.targets[index]))
        domain = self.domains[index]

        if self.transform:
            augmentations = self.transform(image=image, mask=target)
            image, target = augmentations["image"], augmentations["mask"]

        return image, target, domain
