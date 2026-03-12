import torch.nn as nn
import torch
from networks.masker import Masker
from networks.newprojector import Projector
class causal(nn.Module):
    def __init__(self):
        super(causal, self).__init__()
        self.seg_head_causal = nn.Conv2d(32, 2, 1)  
        self.seg_head_confound = nn.Conv2d(32, 2, 1)
        self.masker = Masker(32, 32) 
    def forward(self, img):
        v=img
        mask_sup = self.masker(v) 
        mask_inf = torch.ones_like(mask_sup) - mask_sup 
        v_sup = v * mask_sup
        v_inf = v * mask_inf

        return v_sup, v_inf
