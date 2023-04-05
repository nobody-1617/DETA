import torch.nn as nn
import torch
import numpy as np

"""Reference: Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""
class SupCluLoss(nn.Module):
    def __init__(self, temperature=0.3, contrast_mode='all', base_temperature=0.07):
        super(SupCluLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, shot_nums=None, is_weight_patch=False, q_emb=None, mask=None, K_patch=5):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device) # 200 * 200
        else:
            mask = mask.float().to(device)

        contrast_count = 1
        contrast_feature = features
        if self.contrast_mode == 'one':
            assert False
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T)
        weight_pair = torch.ones(anchor_dot_contrast.shape).cuda()

        """ Ours """
        is_softmax = True
        is_Mean, is_Max = True, False
        way = len(shot_nums)
        N_patch = sum(shot_nums) * K_patch
        if is_weight_patch:
            # Compute weights for patch pairs
            with torch.no_grad():
                q_emb = nn.functional.normalize(q_emb, dim=1)
                simscore = torch.matmul(q_emb, q_emb.T)
                if not is_softmax:
                    relu = nn.ReLU()
                    simscore =relu(simscore)
                else:
                    is_Mean = False

                # positive, without intra-image patches
                simscore_ps = simscore * mask
                for p in range(sum(shot_nums)):
                    simscore_ps[p*K_patch:(p+1)*K_patch, p*K_patch:(p+1)*K_patch] = 0

                # negative
                mask2 = torch.ones(mask.shape).to(device) - mask
                simscore_ng = simscore * mask2

                w_c = []
                point = 0
                temp = 0.5
                e = 0.00001
                for w in range(way):
                    w_ps = simscore_ps[point: point + shot_nums[w] * K_patch].sum(-1) / (K_patch * shot_nums[w] - K_patch + e)
                    w_ng = simscore_ng[point: point + shot_nums[w] * K_patch].sum(-1) / (N_patch - K_patch * shot_nums[w] + e)
                    if is_softmax:
                        w_ps = torch.softmax(w_ps/temp, dim=-1)
                        w_ng = torch.softmax(w_ng/temp, dim=-1)
                        w_ci = w_ps / w_ng
                        # w_ci = w_ci / max(w_ci)
                    else:
                        if is_Mean:
                            w_ps = w_ps / (w_ps.mean() + e)
                            w_ng = w_ng / (w_ng.mean() + e)
                        w_ci = w_ps / (w_ng + e)
                    [w_c.append(i) for i in w_ci]
                    point = point + shot_nums[w]*K_patch

                w_c = torch.stack(w_c)
                if is_Max:
                    w_c = w_c / w_c.max()
                w_c = w_c.unsqueeze(0)
                weight_pair = torch.matmul(w_c.T, w_c)

        # anchor_dot_contrast = torch.div(anchor_dot_contrast, self.temperature)
        anchor_dot_contrast = torch.div(anchor_dot_contrast, self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits = logits * weight_pair # jim
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mask_log_prob = mask * log_prob
        mean_log_prob_pos = mask_log_prob.sum(1) / mask.sum(1)

        # mean_log_prob_pos = w_c.squeeze(0) * mean_log_prob_pos # jim
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss, w_c.T

