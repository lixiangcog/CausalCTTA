from torch import nn
import torch
from networks.resnet import resnet34, resnet18, resnet50, resnet101, resnet152
import torch.nn.functional as F

from .layers import Projector, FPN_AD
class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)

        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        return self.bn(F.relu(cat_p))


class ResUnet(nn.Module):
    def __init__(self, resnet='resnet34', num_classes=2, pretrained=False):
        super().__init__()
        if resnet == 'resnet34':
            base_model = resnet34
            feature_channels = [64, 64, 128, 256, 512]
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
            feature_channels = [64, 256, 512, 1024, 2048]
        elif resnet == 'resnet101':
            base_model = resnet101
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')

        # layers = list(base_model(pretrained=pretrained, IBN=IBN, mixstyle_layers=mixstyle_layers).children())[:8]
        # base_layers = nn.Sequential(*layers)
        # self.res = base_layers
        self.res = base_model(pretrained=pretrained)

        self.num_classes = num_classes

        self.up1 = UnetBlock(feature_channels[4], feature_channels[3], 256)
        self.up2 = UnetBlock(256, feature_channels[2], 256)
        self.up3 = UnetBlock(256, feature_channels[1], 256)
        self.up4 = UnetBlock(256, feature_channels[0], 256)

        self.up5 = nn.ConvTranspose2d(256, 32, 2, stride=2)
        self.bnout = nn.BatchNorm2d(32)
        self.neck_ad = FPN_AD(in_channels=[32, 256, 256], out_channels=[32, 32, 32])

        self.seg_head_causal = nn.Conv2d(32, num_classes, 1)  # 用于因果特征
        self.seg_head_confound = nn.Conv2d(32, num_classes, 1)  # 用于混淆特征
        self.seg_head = nn.Conv2d(32, num_classes, 1)  # 原始分割头

    def forward(self, x):
        x, sfs = self.res(x)
        x = F.relu(x)

        x1 = self.up1(x, sfs[3])
        x2 = self.up2(x1, sfs[2])
        x3 = self.up3(x2, sfs[1])
        x4 = self.up4(x3, sfs[0])
        x5 = self.up5(x4)
        head_input = F.relu(self.bnout(x5))

        fq_sup,fq_inf= self.neck_ad([head_input,x4,x3])

        # 分别生成分割输出
        pred_causal = self.seg_head_causal(fq_sup)
        pred_confound = self.seg_head_confound(fq_inf)
        pred_original = self.seg_head(head_input)

        return pred_causal,pred_confound,pred_original

    def close(self):
        for sf in self.sfs:
            sf.remove()


if __name__ == "__main__":
    model = ResUnet(resnet='resnet34', num_classes=2, pretrained=False)
    print(model.res)
    model.cuda().eval()
    input = torch.rand(2, 3, 512, 512).cuda()
    seg_output, x_iw_list, iw_loss = model(input)
    print(seg_output.size())

