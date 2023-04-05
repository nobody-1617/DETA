import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.losses import prototype_loss
from models.sscl import SupCluLoss
from torchvision import transforms
import random

class projector(nn.Module):
    def __init__(self, z_dim=128, feat_dim=512, type='mlp'):
        super(projector, self).__init__()
        if type == 'mlp':
            self.proj = nn.Sequential(nn.Linear(feat_dim, 512), nn.ReLU(), nn.Linear(512, z_dim))
        else:
            self.proj = nn.Linear(feat_dim, z_dim)

    def forward(self, x):
        x = self.proj(x)
        return x

def resize_transform_tensor(images, size=84):
    transform_ = transforms.Compose([transforms.Resize(size)])
    images_list = []
    [images_list.append(transform_(img)) for img in images]
    return torch.stack(images_list).cuda()


def gen_patches_tensor(context_images, shot_nums, size=84, overlap=0.3, K_patch=5):
    _, _, h, w = context_images.shape
    way = len(shot_nums)
    ch = int(overlap * h)
    cw = int(overlap * w)
    patches = []
    bbx = []

    for _ in range(K_patch):
        bbx.append([h // 4 - ch // 2, w // 4 - cw // 2, 3 * h // 4 + ch // 2, 3 * w // 4 + cw // 2])

    max_x0y0 = h // 2 + ch - size
    for x in range(K_patch):
        patch_x = context_images[:,:, bbx[x][0]:bbx[x][2], bbx[x][1]:bbx[x][3]]
        start_x = random.randint(0, max_x0y0 - 1)
        start_y = random.randint(0, max_x0y0 - 1)
        patch_xx = patch_x[:,:, start_x:start_x+size, start_y:start_y+size]
        patches.append(patch_xx)

    point = 0
    patches_img = []
    for w in range(way):
        patches_class = []
        for p in range(K_patch):
            pat = patches[p][point: point+shot_nums[w]]
            patches_class.append(pat)
        point = point+shot_nums[w]
        for s in range(shot_nums[w]):
            for pt in patches_class:
                patches_img.append(pt[s])
    images_gather = torch.stack(patches_img, dim=0)

    return images_gather


def ta(context_images, context_labels, model, model_name="MOCO", max_iter=40, lr_finetune=0.001, distance='cos',
        is_baseline = False, is_weight_patch=False, is_weight_sample=False, K_patch=5, dataset=""):

    model.eval()
    lr = lr_finetune
    if model_name == 'CLIP':
        feat = model.encode_image(context_images[0].unsqueeze(0))
    else:
        feat = model(context_images[0].unsqueeze(0))
    proj = projector(feat_dim=feat.shape[1]).cuda()
    
    params = []
    backbone_params = [v for k, v in model.named_parameters()]
    params.append({'params': backbone_params})
    proj_params = [v for k, v in proj.named_parameters()]
    params.append({'params': proj_params})

    optimizer = torch.optim.Adadelta(params, lr=lr)
    criterion_clu = SupCluLoss(temperature=0.07)
    shot_nums = []
    shot_nums_sum = 0
    n_way = len(context_labels.unique())
    labels_all = []
    for i in range(n_way):
        ith_way_shotnums = context_images[(context_labels == i).nonzero(), :].shape[0]
        shot_nums.append(ith_way_shotnums)
        shot_nums_sum = shot_nums_sum + ith_way_shotnums
        label_ci = [i] * shot_nums[i] * K_patch
        labels_all = labels_all + label_ci
    label_clu_way = torch.LongTensor(list(np.reshape(labels_all, (1, -1)).squeeze())).cuda()

    balance = 0.1
    START_WEIGHT = 10
    lamb = 0.7
    size_list = [84,128]
    sample_weight = None

    if is_baseline:
        START_WEIGHT = 10086
    for i in range(max_iter):
        optimizer.zero_grad()
        model.zero_grad()

        """ For images """
        if model_name == 'CLIP':
            context_features = model.encode_image(context_images)
        else:
            context_features = model(context_images)
        if len(context_features.shape) == 4:
            avg_pool = nn.AvgPool2d(context_features.shape[-2:])
            context_features = avg_pool(context_features).squeeze(-1).squeeze(-1)

        """ For patches """
        if i >= START_WEIGHT:
            size = random.choice(size_list)
            images_gather = gen_patches_tensor(context_images, shot_nums, size=size, overlap=0.3, K_patch=K_patch)

            if model_name in ['CLIP','DEIT','SWIN']:
                images_gather = resize_transform_tensor(images_gather, size=224)
            if model_name == 'CLIP':
                q_emb = model.encode_image(images_gather.cuda()).float()
            else:
                q_emb = model(images_gather.cuda()).float()
            if len(q_emb.shape) == 4:
                avg_pool = nn.AvgPool2d(q_emb.shape[-2:])
                q_emb = avg_pool(q_emb).squeeze(-1).squeeze(-1)

            q = proj(q_emb)
            q_norm = nn.functional.normalize(q, dim=1)
            loss_1, patch_weight = criterion_clu(q_norm, label_clu_way, shot_nums=shot_nums, is_weight_patch=is_weight_patch, q_emb=q_emb, K_patch=K_patch)

            # compute sample weight of the current iter
            if is_weight_sample:
                patch_w = patch_weight.squeeze(-1)
                sample_weight_i = []
                for s in range(shot_nums_sum):
                    s_w = patch_w[s*K_patch: (s+1)*K_patch].mean()
                    sample_weight_i.append(s_w)
                sample_weight_i = torch.stack(sample_weight_i)
                if i == START_WEIGHT:
                    sample_weight = sample_weight_i
                else:
                    sample_weight = lamb * sample_weight + (1-lamb) * sample_weight_i
                context_features = sample_weight.unsqueeze(-1) * context_features

            loss_2, stat, _ = prototype_loss(context_features, context_labels, q_emb, label_clu_way, patch_weight=patch_weight, distance=distance)
            loss = balance * loss_1 + loss_2

        else:
            loss, stat, _ = prototype_loss(context_features, context_labels, context_features, context_labels, patch_weight=None, distance=distance)

        loss.backward()
        optimizer.step()

    return sample_weight





















