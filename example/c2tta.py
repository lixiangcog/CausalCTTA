import os
import torch
import numpy as np
import argparse, sys, datetime
from config import *
from torch.autograd import Variable
from utils.convert import AdaBN
from utils.memory import Memory
from utils.prompt import Prompt
from utils.metrics import calculate_metrics
from networks.origin_ResUnet_TTA import ResUnet
from torch.utils.data import DataLoader
from dataloaders.OPTIC_dataloader import OPTIC_dataset
from dataloaders.transform import collate_fn_wo_transform
from dataloaders.convert_csv_to_list import convert_labeled_list
from networks.causal import causal
from networks.newprojector import Projector
torch.set_num_threads(1)
import pandas as pd
from PCA import CausalTrimming

import json
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS

class VPTTA:
    def __init__(self, config):
        # Data Loading
        target_test_csv = []
        for target in config.Target_Dataset:
            if target != 'REFUGE_Valid' and target != 'ORIGA':
                target_test_csv.append(target + '_train_pseudo.csv')
                target_test_csv.append(target + '_test_pseudo.csv')
            elif target == 'REFUGE_Valid':
                target_test_csv.append(target + '_pseudo.csv')
            else:
                target_test_csv.append(target + '_train_pseudo.csv')   

        ts_img_list = list()
        ts_label_list = list()
        ts_pseudo_list = list()
        for csv_file in target_test_csv:
            data = pd.read_csv(os.path.join(config.dataset_root, csv_file))
            ts_img_list += data['image'].tolist()
            ts_label_list += data['mask'].tolist()
            ts_pseudo_list += data['pseudo_label'].tolist()
        print("The number of target test samples: ", len(ts_img_list))
        target_test_dataset = OPTIC_dataset(config.dataset_root, ts_img_list, ts_label_list, ts_pseudo_list,
                                            config.image_size, img_normalize=True, Training=False)
        self.target_test_loader = DataLoader(dataset=target_test_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             drop_last=False,
                                             collate_fn=collate_fn_wo_transform,
                                             num_workers=config.num_workers)
        self.image_size = config.image_size

        # Model
        self.load_model = os.path.join(config.model_root,str(config.Source_Dataset))  # Pre-trained Source Model
        self.backbone = config.backbone
        self.in_ch = config.in_ch
        self.out_ch = config.out_ch

        # Optimizer
        self.optim = config.optimizer
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.momentum = config.momentum
        self.betas = (config.beta1, config.beta2)

        # GPU
        self.device = config.device

        self.lossmap = ['dice', 'bce']
        self.seg_cost = Seg_loss(self.lossmap)

        # Warm-up
        self.warm_n = config.warm_n

        # Prompt
        self.prompt_alpha = config.prompt_alpha
        self.iters = config.iters

        self.use_prompt = config.use_prompt
        self.use_vida = config.use_vida
        self.use_dropout = config.use_dropout
        self.convert = config.use_AdaBN  # Always convert to AdaBN
        self.use_trans_input= config.use_trans_input
        self.clip= config.clip
        # Initialize the pre-trained model and optimizer
        self.build_model()

        # Memory Bank
        self.neighbor = config.neighbor
        self.memory_bank = Memory(size=config.memory_size, dimension=self.prompt.data_prompt.numel())

        # Print Information
        for arg, value in vars(config).items():
            print(f"{arg}: {value}")
        self.print_prompt()
        print('***' * 20)

    def build_model(self):
        self.prompt = Prompt(prompt_alpha=self.prompt_alpha, image_size=self.image_size).to(self.device)
        self.causal_module = causal().to(self.device)
        self.projector_causal = Projector(word_dim=512).to(self.device)
        self.projector_confound = Projector(word_dim=512).to(self.device)
        self.model = ResUnet(resnet=self.backbone, num_classes=self.out_ch, pretrained=False, convert=self.convert,newBN=AdaBN, warm_n=self.warm_n).to(self.device)
        checkpoint = torch.load(os.path.join(self.load_model, 'last-Res_Unet.pth'), map_location=self.device)
        self.model.load_state_dict(checkpoint, strict=True)
        if self.clip:
            model_name = "biomedclip_local"
            with open("VPTTA/checkpoints/open_clip_config.json", "r") as f:
                config = json.load(f)
                model_cfg = config["model_cfg"]
                preprocess_cfg = config["preprocess_cfg"]
            if (not model_name.startswith(HF_HUB_PREFIX)
                and model_name not in _MODEL_CONFIGS
                and config is not None):
                _MODEL_CONFIGS[model_name] = model_cfg
            self.tokenizer = get_tokenizer(model_name)
            clip_model, _, preprocess = create_model_and_transforms(
                model_name=model_name,
                pretrained=None,
            )
            checkpoint_path = "VPTTA/checkpoints/open_clip_pytorch_model.bin"
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            clip_model.load_state_dict(checkpoint, strict=True)
            self.text_encoder = clip_model.text.to(self.device)

        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.prompt.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                nesterov=True,
                weight_decay=self.weight_decay
            )
            params_to_optimize = []
            params_to_optimize.extend(self.causal_module.parameters())
            if self.clip:
                params_to_optimize.extend(self.projector_causal.parameters())
                params_to_optimize.extend(self.projector_confound.parameters())
            self.optimizer2 = torch.optim.SGD(
                params_to_optimize,
                lr=self.lr,
                momentum=self.momentum,
                nesterov=True,
                weight_decay=self.weight_decay
            )
                
        elif self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.prompt.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay
            )
            params_to_optimize = []
            params_to_optimize.extend(self.causal_module.parameters())
            if self.clip:
                params_to_optimize.extend(self.projector_causal.parameters())
                params_to_optimize.extend(self.projector_confound.parameters())
            self.optimizer2 = torch.optim.Adam(
                params_to_optimize,
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay
            )


    def print_prompt(self):
        num_params = 0
        if self.use_prompt:
            for p in self.prompt.parameters():
                num_params += p.numel()
        for p in self.causal_module.parameters():
            num_params += p.numel()
        if self.clip:
            for p in self.projector_causal.parameters():
                num_params += p.numel()
            for p in self.projector_confound.parameters():
                num_params += p.numel()
        print("The number of total parameters: {}".format(num_params))

    def run(self):
        metric_dict = ['Disc_Dice', 'Disc_ASD', 'Cup_Dice', 'Cup_ASD']

        # Valid on Target
        metrics_test = [[], [], [], []]

        for batch, data in enumerate(self.target_test_loader):
            newdata, y, pseudo = data['data'], data['mask'], data['pseudo_label']
            newdata = np.array(newdata)   # 如果x还不是数组，先转换
            newdata = newdata.transpose(1,0,2,3,4)  # 调整维度顺序
            
            newdata = torch.from_numpy(newdata).to(dtype=torch.float32)
            y = torch.from_numpy(y).to(dtype=torch.float32)
            pseudo = torch.from_numpy(pseudo).to(dtype=torch.float32)

            newdata, y, pseudo = Variable(newdata).to(self.device), Variable(y).to(self.device), Variable(pseudo).to(self.device)

            x=newdata[0,:,:,:,:]  # 只使用原始图像进行测试

            if self.use_prompt:
                self.model.eval()
                self.prompt.train()
                self.causal_module.train()
                self.model.change_BN_status(new_sample=True)
                if len(self.memory_bank.memory.keys()) >= self.neighbor:
                    _, low_freq = self.prompt(x)
                    init_data, score = self.memory_bank.get_neighbours(keys=low_freq.cpu().numpy(), k=self.neighbor)
                else:
                    init_data = torch.ones((1, 3, self.prompt.prompt_size, self.prompt.prompt_size)).data
                self.prompt.update(init_data)

                # Train Prompt for n iters (1 iter in our VPTTA)
                for tr_iter in range(self.iters):
                    prompt_x, _ = self.prompt(x)
                    _,_,_,imgs = self.model(prompt_x)
                    times, bn_loss = 0, 0
                    for nm, m in self.model.named_modules():
                        if isinstance(m, AdaBN):
                            bn_loss += m.bn_loss
                            times += 1
                    v_sup, v_inf = self.causal_module(imgs)
                    if self.clip:
                        text_features = self.text_encoder(self.tokenizer(["Optic disc in fundus photography"]).to(self.device))
                        pred_causal= self.projector_causal(v_sup, text_features)
                        pred_confound= self.projector_confound(v_inf, text_features)
                    else:
                        pred_causal= self.causal_module.seg_head_causal(v_sup)
                        pred_confound= self.causal_module.seg_head_confound(v_inf)
                    loss_causal = self.seg_cost(pred_causal, pseudo)
                    loss_confound = self.seg_cost(pred_confound, pseudo)
                    bn_loss = bn_loss / times
                    loss = (loss_causal - 0.05 * loss_confound)/1000+bn_loss
                    
                    self.optimizer.zero_grad()
                    self.optimizer2.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer2.step()

                    self.model.change_BN_status(new_sample=False)

                # Inference
                self.model.eval()
                self.prompt.eval()
                self.causal_module.eval()
            
                with torch.no_grad():
                    prompt_x, low_freq = self.prompt(x)
                    _,_,_,imgs = self.model(prompt_x)
                    v_sup, v_inf = self.causal_module(imgs)
                    if self.clip:
                        text_features = self.text_encoder(self.tokenizer(["Optic disc in fundus photography"]).to(self.device))
                        pred_causal= self.projector_causal(v_sup, text_features)
                        pred_confound= self.projector_confound(v_inf, text_features)
                    else:
                        pred_causal= self.causal_module.seg_head_causal(v_sup)
                        pred_confound= self.causal_module.seg_head_confound(v_inf)
                # Update the Memory Bank
                self.memory_bank.push(keys=low_freq.cpu().numpy(), logits=self.prompt.data_prompt.detach().cpu().numpy())

            elif self.convert:
                # Inference without prompt
                self.prompt.eval()
                self.model.eval()
                self.causal_module.train()

                self.model.change_BN_status(new_sample=True)
                for tr_iter in range(self.iters):
                    prompt_x, _ = self.prompt(x)
                    _,_,_,imgs = self.model(prompt_x)
                    times, bn_loss = 0, 0
                    for nm, m in self.model.named_modules():
                        if isinstance(m, AdaBN):
                            bn_loss += m.bn_loss
                            times += 1
                    v_sup, v_inf = self.causal_module(imgs)
                    if self.clip:
                        text_features = self.text_encoder(self.tokenizer(["Optic disc in fundus photography"]).to(self.device))
                        pred_causal= self.projector_causal(v_sup, text_features)
                        pred_confound= self.projector_confound(v_inf, text_features)
                    else:
                        pred_causal= self.causal_module.seg_head_causal(v_sup)
                        pred_confound= self.causal_module.seg_head_confound(v_inf)

                    loss_causal = self.seg_cost(pred_causal, pseudo)
                    loss_confound = self.seg_cost(pred_confound, pseudo)

                    bn_loss = bn_loss / times
                    loss = (loss_causal - 0.05 * loss_confound)/1000+bn_loss
                    
                    self.optimizer2.zero_grad()
                    loss.backward()
                    self.optimizer2.step()

                    self.model.change_BN_status(new_sample=False)
                    with torch.no_grad():
                        prompt_x, low_freq = self.prompt(x)
                        _,_,_,imgs = self.model(prompt_x)
                        v_sup, v_inf = self.causal_module(imgs)
                        if self.clip:
                            text_features = self.text_encoder(self.tokenizer(["Optic disc in fundus photography"]).to(self.device))
                            pred_causal= self.projector_causal(v_sup, text_features)
                            pred_confound= self.projector_confound(v_inf, text_features)
                        else:
                            pred_causal= self.causal_module.seg_head_causal(v_sup)
                            pred_confound= self.causal_module.seg_head_confound(v_inf)
            else:
                self.prompt.eval()
                self.model.eval()
                self.causal_module.train()

                for tr_iter in range(1):
                    prompt_x, _ = self.prompt(x)
                    _,_,_,imgs = self.model(prompt_x)
                    v_sup, v_inf = self.causal_module(imgs)
                    if self.clip:
                        text_features = self.text_encoder(self.tokenizer(["Optic disc in fundus photography"]).to(self.device))
                        pred_causal= self.projector_causal(v_sup, text_features)
                        pred_confound= self.projector_confound(v_inf, text_features)
                    else:
                        pred_causal= self.causal_module.seg_head_causal(v_sup)
                        pred_confound= self.causal_module.seg_head_confound(v_inf)
                    loss_causal = self.seg_cost(pred_causal, pseudo)
                    loss_confound = self.seg_cost(pred_confound, pseudo)
                    loss = (loss_causal - 0.05 * loss_confound)/1000
                    
                    self.optimizer2.zero_grad()
                    loss.backward()
                    self.optimizer2.step()
                    with torch.no_grad():
                        prompt_x, low_freq = self.prompt(x)
                        _,_,_,imgs = self.model(prompt_x)
                        v_sup, v_inf = self.causal_module(imgs)
                        if self.clip:
                            text_features = self.text_encoder(self.tokenizer(["Optic disc in fundus photography"]).to(self.device))
                            pred_causal= self.projector_causal(v_sup, text_features)
                            pred_confound= self.projector_confound(v_inf, text_features)
                        else:
                            pred_causal= self.causal_module.seg_head_causal(v_sup)
                            pred_confound= self.causal_module.seg_head_confound(v_inf)

            # Calculate the metrics
            import matplotlib.pyplot as plt
            from PIL import Image
            seg_output = torch.sigmoid(pred_causal)
            metrics = calculate_metrics(seg_output.detach().cpu(), y.detach().cpu())
            for i in range(len(metrics)):
                assert isinstance(metrics[i], list), "The metrics value is not list type."
                metrics_test[i] += metrics[i]

        test_metrics_y = np.mean(metrics_test, axis=1)
        print_test_metric_mean = {}
        for i in range(len(test_metrics_y)):
            print_test_metric_mean[metric_dict[i]] = test_metrics_y[i]
        print("Test Metrics: ", print_test_metric_mean)
        print('Mean Dice:', (print_test_metric_mean['Disc_Dice'] + print_test_metric_mean['Cup_Dice']) / 2)
        line1 = "Test Metrics: " + str(print_test_metric_mean)
        line2 = "Mean Dice: " + str((print_test_metric_mean['Disc_Dice'] + print_test_metric_mean['Cup_Dice']) / 2)
        with open("results.txt", "a") as f:
            f.write(line1 + "\n")
            f.write(line2 + "\n")
            f.write("\n")  # 写入一个空行，作为每次运行的分隔


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--Source_Dataset', type=str, default='RIM_ONE_r3',
                        help='RIM_ONE_r3/REFUGE/ORIGA/REFUGE_Valid/Drishti_GS')
    parser.add_argument('--Target_Dataset', type=list)

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=512)

    # Model
    parser.add_argument('--backbone', type=str, default='resnet34', help='resnet34/resnet50')
    parser.add_argument('--in_ch', type=int, default=3)
    parser.add_argument('--out_ch', type=int, default=2)

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='Adam', help='SGD/Adam')
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--momentum', type=float, default=0.99)  # momentum in SGD
    parser.add_argument('--beta1', type=float, default=0.9)      # beta1 in Adam
    parser.add_argument('--beta2', type=float, default=0.99)     # beta2 in Adam.
    parser.add_argument('--weight_decay', type=float, default=0.00)

    # Training
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--iters', type=int, default=1)

    # Hyperparameters in memory bank, prompt, and warm-up statistics
    parser.add_argument('--memory_size', type=int, default=40)
    parser.add_argument('--neighbor', type=int, default=16)
    parser.add_argument('--prompt_alpha', type=float, default=0.01)
    parser.add_argument('--warm_n', type=int, default=5)

    # Path
    parser.add_argument('--path_save_log', type=str, default='./logs')
    parser.add_argument('--model_root', type=str, default='./models')
    parser.add_argument('--dataset_root', type=str, default='/media/userdisk0/zychen/Datasets/Fundus')

    # Cuda (default: the first available device)
    parser.add_argument('--device', type=str, default='cuda:0')

    # Decide whether to use the prompt
    parser.add_argument('--use_prompt', action='store_true', help='Whether to use the prompt')
    parser.add_argument('--use_AdaBN', action='store_true', help='Whether to convert BN to AdaBN')
    parser.add_argument('--clip', action='store_true', help='Whether to use CLIP text encoder')

    config = parser.parse_args()

    config.Target_Dataset = ['RIM_ONE_r3', 'REFUGE', 'ORIGA', 'REFUGE_Valid', 'Drishti_GS']
    config.Target_Dataset.remove(config.Source_Dataset)

    TTA = VPTTA(config)
    TTA.run()
