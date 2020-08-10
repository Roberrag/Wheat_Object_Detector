import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionLoss(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + OHEMLoss(cls_preds, cls_targets).
        '''

        ################################################################
        # loc_loss
        ################################################################

        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.long().sum(1, keepdim=True)

        mask = pos.unsqueeze(2).expand_as(loc_preds)  # [N,#anchors,4]
        masked_loc_preds = loc_preds[mask].view(-1, 4)  # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1, 4)  # [#pos,4]
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, reduction='none')
        loc_loss = loc_loss.sum() / num_pos.sum().float()

        ################################################################
        # cls_loss with OHEM
        ################################################################

        # Compute max conf across batch for hard negative mining
        batch_size, _ = cls_targets.size()
        batch_conf = cls_preds.view(-1, self.num_classes)
        cls_loss = F.cross_entropy(batch_conf, cls_targets.view(-1), ignore_index=-1, reduction='none')
        cls_loss = cls_loss.view(batch_size, -1)

        # Hard Negative Mining
        # filter out pos boxes (pos = cls_targets > 0) for now.
        pos_cls_loss = cls_loss[pos]


        cls_loss[pos] = 0


        _, loss_idx = cls_loss.sort(1, descending=True)


        _, idx_rank = loss_idx.sort(1)



        negpos_ratio = 3



        num_neg = torch.clamp(negpos_ratio * num_pos, min=1, max=pos.size(1) - 1)

        neg = idx_rank < num_neg.expand_as(idx_rank)
        neg_cls_loss = cls_loss[neg]

        cls_loss = (pos_cls_loss.sum() + neg_cls_loss.sum()) / num_pos.sum().float()

        # The magnitude of cross-entropy loss is much more than L2, L1, and smooth L1.
        # So it is better to take a weighted loss. Here we have chosen twenty times of
        # lower magnitude loss and one time of higher magnitude loss.

        return 20 * loc_loss, cls_loss
