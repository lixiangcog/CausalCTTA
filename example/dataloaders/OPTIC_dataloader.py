import math
from torch.utils import data
import numpy as np
from PIL import Image
from batchgenerators.utilities.file_and_folder_operations import *
from dataloaders.normalize import normalize_image, normalize_image_to_0_1
import matplotlib.pyplot as plt
from augmentation import OCTAugmentor

class OPTIC_dataset(data.Dataset):
    def __init__(self, root, img_list, label_list, pseudo_list, target_size=512, batch_size=None, img_normalize=True, Training=True):
        super().__init__()
        self.root = root
        self.img_list = img_list
        self.label_list = label_list
        self.pseudo_list = pseudo_list
        self.len = len(img_list)
        self.target_size = (target_size, target_size)
        self.img_normalize = img_normalize
        self.Training = Training
        # if batch_size is not None:
        #     iter_nums = len(self.img_list) // batch_size
        #     scale = math.ceil(100 / iter_nums)
        #     self.img_list = self.img_list * scale
        #     self.label_list = self.label_list * scale

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        if self.label_list[item].endswith('tif'):
            self.label_list[item] = self.label_list[item].replace('.tif', '-{}.tif'.format(1))
        img_file = os.path.join(self.root, self.img_list[item])
        label_file = os.path.join(self.root, self.label_list[item])
        pseudo_file = self.pseudo_list[item]
        img = Image.open(img_file)
        label = Image.open(label_file).convert('L')
        pseudo_label = Image.open(pseudo_file).convert('L')

        img = img.resize(self.target_size)
        imglist = [img]
        aug = OCTAugmentor()
        for _ in range(15):
            imglist.append(aug(img=np.array(img)))

        label = label.resize(self.target_size, resample=Image.NEAREST)
        pseudo_label = pseudo_label.resize(self.target_size, resample=Image.NEAREST)

        for _ in range(len(imglist)):
            imglist[_] = np.array(imglist[_]).transpose(2, 0, 1).astype(np.float32)
            if self.img_normalize:
                imglist[_] = normalize_image_to_0_1(imglist[_])

        label_npy = np.array(label)
        pseudo = np.array(pseudo_label)

        mask = np.zeros_like(label_npy)
        mask[label_npy < 255] = 1
        mask[label_npy == 0] = 2

        if self.Training:
            return imglist[0], mask[np.newaxis], pseudo[np.newaxis], img_file
        else:
            return imglist, mask[np.newaxis], pseudo[np.newaxis], img_file

