#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
from .functions import bbox_iou, BboxType


__all__ = ["YoloLayer"]


class YoloLayer(nn.Module):
    def __init__(
        self,
        anchors_all,
        anchors_mask,
        num_classes,
        lambda_xy=1,
        lambda_wh=1,
        lambda_conf=1,
        lambda_cls=1,
        obj_scale=1,
        noobj_scale=1,
        ignore_thres=0.7,
        epsilon=1e-16,
    ):
        super(YoloLayer, self).__init__()

        assert num_classes > 0

        self._anchors_all = anchors_all
        self._anchors_mask = anchors_mask

        self._num_classes = num_classes
        self._bbox_attrib = 5 + num_classes

        self._lambda_xy = lambda_xy
        self._lambda_wh = lambda_wh
        self._lambda_conf = lambda_conf

        if self._num_classes == 1:
            self._lambda_cls = 0
        else:
            self._lambda_cls = lambda_cls

        self._obj_scale = obj_scale
        self._noobj_scale = noobj_scale
        self._ignore_thres = ignore_thres

        self._epsilon = epsilon

        self._mseloss = nn.MSELoss(reduction="sum")
        self._bceloss = nn.BCELoss(reduction="sum")
        self._bceloss_average = nn.BCELoss(reduction="elementwise_mean")

    def forward(self, x: torch.Tensor, img_dim: tuple, target=None):
        # x : batch_size * nA * (5 + num_classes) * H * W

        device = x.device
        if target is not None:
            assert target.device == x.device

        nB = x.shape[0]
        nA = len(self._anchors_mask)
        nH, nW = x.shape[2], x.shape[3]
        stride = img_dim[1] / nH
        anchors_all = torch.FloatTensor(self._anchors_all) / stride
        anchors = anchors_all[self._anchors_mask]

        # Reshape predictions from [B x [A * (5 + num_classes)] x H x W] to [B x A x H x W x (5 + num_classes)]
        preds = x.view(nB, nA, self._bbox_attrib, nH, nW).permute(0, 1, 3, 4, 2).contiguous()

        # tx, ty, tw, wh
        preds_xy = preds[..., :2].sigmoid()
        preds_wh = preds[..., 2:4]
        preds_conf = preds[..., 4].sigmoid()
        preds_cls = preds[..., 5:].sigmoid()

        # calculate cx, cy, anchor mesh
        mesh_y, mesh_x = torch.meshgrid([torch.arange(nH, device=device), torch.arange(nW, device=device)])
        mesh_xy = torch.stack((mesh_x, mesh_y), 2).float()

        mesh_anchors = anchors.view(1, nA, 1, 1, 2).repeat(1, 1, nH, nW, 1).to(device)

        # pred_boxes holds bx,by,bw,bh
        pred_boxes = torch.FloatTensor(preds[..., :4].shape)
        pred_boxes[..., :2] = preds_xy + mesh_xy
        pred_boxes[..., 2:4] = preds_wh.exp() * mesh_anchors

        if target is not None:
            (
                obj_mask,
                noobj_mask,
                box_coord_mask,
                tconf,
                tcls,
                tx,
                ty,
                tw,
                th,
                nCorrect,
                nGT,
            ) = self.build_target_tensor(
                pred_boxes, target, anchors_all, anchors, (nH, nW), self._num_classes, self._ignore_thres,
            )

            # masks for loss calculations
            obj_mask, noobj_mask = obj_mask.to(device), noobj_mask.to(device)
            box_coord_mask = box_coord_mask.to(device)
            cls_mask = obj_mask == 1
            tconf, tcls = tconf.to(device), tcls.to(device)
            tx, ty, tw, th = tx.to(device), ty.to(device), tw.to(device), th.to(device)

            loss_x = self._lambda_xy * self._mseloss(preds_xy[..., 0] * box_coord_mask, tx * box_coord_mask) / 2
            loss_y = self._lambda_xy * self._mseloss(preds_xy[..., 1] * box_coord_mask, ty * box_coord_mask) / 2
            loss_w = self._lambda_wh * self._mseloss(preds_wh[..., 0] * box_coord_mask, tw * box_coord_mask) / 2
            loss_h = self._lambda_wh * self._mseloss(preds_wh[..., 1] * box_coord_mask, th * box_coord_mask) / 2

            loss_conf = (
                self._lambda_conf
                * (
                    self._obj_scale * self._bceloss(preds_conf * obj_mask, obj_mask)
                    + self._noobj_scale * self._bceloss(preds_conf * noobj_mask, noobj_mask * 0)
                )
                / 1
            )
            loss_cls = self._lambda_cls * self._bceloss(preds_cls[cls_mask], tcls[cls_mask]) / 1
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return (
                loss,
                loss.item() / nB,
                loss_x.item() / nB,
                loss_y.item() / nB,
                loss_w.item() / nB,
                loss_h.item() / nB,
                loss_conf.item() / nB,
                loss_cls.item() / nB,
                nCorrect,
                nGT,
            )

        out = torch.cat((pred_boxes.to(device) * stride, preds_conf.to(device).unsqueeze(4), preds_cls.to(device),), 4,)

        # Reshape predictions from [B x A x H x W x (5 + num_classes)] to [B x [A x H x W] x (5 + num_classes)]
        out = out.permute(0, 2, 3, 1, 4).contiguous().view(nB, nA * nH * nW, self._bbox_attrib)

        return out

    def build_target_tensor(
        self, pred_boxes, target, anchors_all, anchors, inp_dim, num_classes, ignore_thres,
    ):
        nB = target.shape[0]
        nA = len(anchors)
        nH, nW = inp_dim[0], inp_dim[1]
        nCorrect = 0
        nGT = 0
        target = target.float()

        obj_mask = torch.zeros(nB, nA, nH, nW, requires_grad=False)
        noobj_mask = torch.ones(nB, nA, nH, nW, requires_grad=False)
        box_coord_mask = torch.zeros(nB, nA, nH, nW, requires_grad=False)
        tconf = torch.zeros(nB, nA, nH, nW, requires_grad=False)
        tcls = torch.zeros(nB, nA, nH, nW, num_classes, requires_grad=False)
        tx = torch.zeros(nB, nA, nH, nW, requires_grad=False)
        ty = torch.zeros(nB, nA, nH, nW, requires_grad=False)
        tw = torch.zeros(nB, nA, nH, nW, requires_grad=False)
        th = torch.zeros(nB, nA, nH, nW, requires_grad=False)

        for b in range(nB):
            for t in range(target.shape[1]):

                # ignore padded labels
                if target[b, t].sum() == 0:
                    break

                gx = target[b, t, 0] * nW
                gy = target[b, t, 1] * nH
                gw = target[b, t, 2] * nW
                gh = target[b, t, 3] * nH
                gi = int(gx)
                gj = int(gy)

                # pred_boxes - [A x H x W x 4]
                # Do not train for objectness(noobj) if anchor iou > threshold.
                tmp_gt_boxes = torch.FloatTensor([gx, gy, gw, gh]).unsqueeze(0)
                tmp_pred_boxes = pred_boxes[b].view(-1, 4)
                tmp_ious, _ = torch.max(bbox_iou(tmp_pred_boxes, tmp_gt_boxes, bbox_mode=BboxType.CXCYWH), 1)
                ignore_idx = (tmp_ious > ignore_thres).view(nA, nH, nW)
                noobj_mask[b][ignore_idx] = 0

                # find best fit anchor for each ground truth box
                tmp_gt_boxes = torch.FloatTensor([[0, 0, gw, gh]])
                tmp_anchor_boxes = torch.cat((torch.zeros(len(anchors_all), 2), anchors_all), 1)
                tmp_ious = bbox_iou(tmp_anchor_boxes, tmp_gt_boxes, bbox_mode=BboxType.CXCYWH)
                best_anchor = torch.argmax(tmp_ious, 0).item()

                # If the best_anchor belongs to this yolo_layer
                if best_anchor in self._anchors_mask:
                    best_anchor = self._anchors_mask.index(best_anchor)
                    # find iou for best fit anchor prediction box against the ground truth box
                    tmp_gt_box = torch.FloatTensor([gx, gy, gw, gh]).unsqueeze(0)
                    tmp_pred_box = pred_boxes[b, best_anchor, gj, gi].view(-1, 4)
                    tmp_iou = bbox_iou(tmp_gt_box, tmp_pred_box, bbox_mode=BboxType.CXCYWH)

                    if tmp_iou > 0.5:
                        nCorrect += 1

                    # larger gradient for small objects
                    box_coord_mask[b, best_anchor, gj, gi] = math.sqrt(2 - target[b, t, 2] * target[b, t, 3])

                    obj_mask[b, best_anchor, gj, gi] = 1
                    tconf[b, best_anchor, gj, gi] = 1
                    tcls[b, best_anchor, gj, gi, int(target[b, t, 4])] = 1
                    tx[b, best_anchor, gj, gi] = gx - gi
                    ty[b, best_anchor, gj, gi] = gy - gj
                    tw[b, best_anchor, gj, gi] = torch.log(gw / anchors[best_anchor, 0] + self._epsilon)
                    th[b, best_anchor, gj, gi] = torch.log(gh / anchors[best_anchor, 1] + self._epsilon)

                    nGT += 1
        return (
            obj_mask,
            noobj_mask,
            box_coord_mask,
            tconf,
            tcls,
            tx,
            ty,
            tw,
            th,
            nCorrect,
            nGT,
        )
