import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import *


def build_targets(pred_corners, target, num_keypoints, num_anchors, num_classes, nH, nW,
                  noobject_scale, object_scale, sil_thresh, dist_thresh, seen, im_width, im_height):
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    conf_mask   = torch.ones(nB, nA, nH, nW) * noobject_scale
    coord_mask  = torch.zeros(nB, nA, nH, nW)
    cls_mask    = torch.zeros(nB, nA, nH, nW)
    txs = list()
    tys = list()
    for i in range(num_keypoints):
        txs.append(torch.zeros(nB, nA, nH, nW))
        tys.append(torch.zeros(nB, nA, nH, nW))
    tconf = torch.zeros(nB, nA, nH, nW)
    tcls  = torch.zeros(nB, nA, nH, nW)

    num_labels = 2 * num_keypoints + 3  # +2 for width, height and +1 for class within label files
    nAnchors = nA*nH*nW
    nPixels  = nH*nW

    """ conf_mask for no_object scale """
    for b in range(nB):
        cur_pred_corners = pred_corners[b*nAnchors:(b+1)*nAnchors].t()  # shape (2*num_keypoints, nA*nH*nW)
        cur_confs = torch.zeros(nAnchors)
        for t in range(50):
            if target[b][t*num_labels+1] == 0:  # centroid x value == 0
                break
            g = list()
            for i in range(num_keypoints):
                g.append(target[b][t*num_labels+2*i+1])
                g.append(target[b][t*num_labels+2*i+2])

            cur_gt_corners = torch.FloatTensor(g).repeat(nAnchors, 1).t()  # 18 x nAnchors
            cur_confs = torch.max(
                cur_confs,
                corner_confidences(cur_pred_corners, cur_gt_corners, th=dist_thresh, im_width=im_width, im_height=im_height)
            ).view_as(conf_mask[b])  # some irrelevant areas are filtered, in the same grid multiple anchor boxes might exceed the threshold
        conf_mask[b][(cur_confs > sil_thresh).view(nA, nH, nW)] = 0
        # conf_mask[b][cur_confs>sil_thresh] = 0

    """ nGT, nCorrect, coord_mask, cls_mask, txs, tys, tconf, tcls """
    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(50):
            if target[b][t*num_labels+1] == 0:  # centroid x value == 0
                break
            # Get gt box for the current label
            nGT += 1
            gx = list()
            gy = list()
            gt_box = list()
            for i in range(num_keypoints):
                gt_box.extend([target[b][t*num_labels+2*i+1], target[b][t*num_labels+2*i+2]])
                gx.append(target[b][t*num_labels+2*i+1] * nW)
                gy.append(target[b][t*num_labels+2*i+2] * nH)
                if i == 0:
                    gi0  = int(gx[i])
                    gj0  = int(gy[i])
            # Update masks
            best_n = 0  # 1 anchor box
            pred_box = pred_corners[b*nAnchors+best_n*nPixels+gj0*nW+gi0]
            conf = corner_confidence(gt_box, pred_box, th=dist_thresh, im_width=im_width, im_height=im_height)
            coord_mask[b][best_n][gj0][gi0] = 1
            cls_mask[b][best_n][gj0][gi0]   = 1
            conf_mask[b][best_n][gj0][gi0]  = object_scale
            # Update targets
            for i in range(num_keypoints):
                txs[i][b][best_n][gj0][gi0] = gx[i] - gi0
                tys[i][b][best_n][gj0][gi0] = gy[i] - gj0
            tconf[b][best_n][gj0][gi0]      = conf
            tcls[b][best_n][gj0][gi0]       = target[b][t*num_labels]
            # Update recall during training
            if conf > 0.5:
                nCorrect = nCorrect + 1

    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, txs, tys, tconf, tcls


class RegionLoss(nn.Module):
    def __init__(self,
                 num_keypoints=9,
                 num_classes=1,
                 anchors=[],
                 num_anchors=1,
                 coord_scale=1.,
                 object_scale=5.,
                 noobject_scale=0.1,
                 class_scale=1.,
                 conf_silent_thresh=0.6,
                 conf_dist_thresh=30,
                 seen=0,
                 im_width=640,
                 im_height=480,
                 pretrain_num_epochs=15):
        # Define the loss layer
        super(RegionLoss, self).__init__()
        self.num_classes         = num_classes
        self.num_anchors         = num_anchors  # for single object pose estimation, there is only 1 trivial predictor (anchor)
        self.num_keypoints       = num_keypoints
        self.anchors             = anchors
        self.coord_scale         = coord_scale
        self.noobject_scale      = noobject_scale
        self.object_scale        = object_scale
        self.class_scale         = class_scale
        self.conf_silent_thresh  = conf_silent_thresh
        self.conf_dist_thresh    = conf_dist_thresh
        self.seen                = seen
        self.im_width            = im_width
        self.im_height           = im_height
        self.pretrain_num_epochs = pretrain_num_epochs

    def forward(self, output, target, epoch):
        # Parameters
        t0 = time.time()
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)
        num_keypoints = self.num_keypoints

        # Activation
        output = output.view(nB, nA, (num_keypoints*2+1+nC), nH, nW)
        x = list()
        y = list()
        x.append(torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW)))
        y.append(torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW)))
        for i in range(1, num_keypoints):
            x.append(output.index_select(2, Variable(torch.cuda.LongTensor([2 * i + 0]))).view(nB, nA, nH, nW))
            y.append(output.index_select(2, Variable(torch.cuda.LongTensor([2 * i + 1]))).view(nB, nA, nH, nW))
        conf   = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([2 * num_keypoints]))).view(nB, nA, nH, nW))
        cls    = output.index_select(2, Variable(torch.linspace(2*num_keypoints+1, 2*num_keypoints+1+nC-1, nC).long().cuda()))
        cls    = cls.view(nB*nA, nC, nH*nW).transpose(1, 2).contiguous().view(nB*nA*nH*nW, nC)
        t1     = time.time()

        # Create pred boxes
        pred_corners = torch.cuda.FloatTensor(2*num_keypoints, nB*nA*nH*nW)
        grid_x = torch.linspace(0, nW - 1, nW).repeat(nH, 1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        for i in range(num_keypoints):
            pred_corners[2 * i + 0]  = (x[i].data.view_as(grid_x) + grid_x) / nW
            pred_corners[2 * i + 1]  = (y[i].data.view_as(grid_y) + grid_y) / nH
        gpu_matrix = pred_corners.transpose(0, 1).contiguous().view(-1, 2*num_keypoints)
        pred_corners = convert2cpu(gpu_matrix)
        t2 = time.time()

        # Build targets
        nGT, nCorrect, coord_mask, conf_mask, cls_mask, txs, tys, tconf, tcls = \
            build_targets(pred_corners, target.data, num_keypoints, nA, nC, nH, nW,
                          self.noobject_scale, self.object_scale,
                          self.conf_silent_thresh, self.conf_dist_thresh,
                          self.seen, self.im_width, self.im_height)
        cls_mask   = (cls_mask == 1)
        nProposals = int((conf > 0.25).sum().data.item())
        for i in range(num_keypoints):
            txs[i] = Variable(txs[i].cuda())
            tys[i] = Variable(tys[i].cuda())
        tconf      = Variable(tconf.cuda())
        tcls       = Variable(tcls[cls_mask].long().cuda())
        coord_mask = Variable(coord_mask.cuda())
        conf_mask  = Variable(conf_mask.cuda().sqrt())
        cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1, nC).cuda())
        cls        = cls[cls_mask].view(-1, nC)
        t3 = time.time()

        # Create loss
        loss_xs   = list()
        loss_ys   = list()
        for i in range(num_keypoints):
            loss_xs.append(self.coord_scale * nn.MSELoss(reduction='sum')(x[i]*coord_mask, txs[i]*coord_mask)/2.0)
            loss_ys.append(self.coord_scale * nn.MSELoss(reduction='sum')(y[i]*coord_mask, tys[i]*coord_mask)/2.0)
        loss_conf  = nn.MSELoss(reduction='sum')(conf*conf_mask, tconf*conf_mask)/2.0
        loss_x = torch.stack(loss_xs).sum()
        loss_y = torch.stack(loss_ys).sum()

        if epoch > self.pretrain_num_epochs:
            loss  = loss_x + loss_y + loss_conf  # in single object pose estimation, there is no classification loss
        else:
            # pretrain initially without confidence loss
            # once the coordinate predictions get better, start training for confidence as well
            loss  = loss_x + loss_y

        t4 = time.time()

        if False:
            print('-----------------------------------')
            print('        activation : %f' % (t1 - t0))
            print(' create pred_corners : %f' % (t2 - t1))
            print('     build targets : %f' % (t3 - t2))
            print('       create loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))

        print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, conf %f, total %f' % (self.seen, nGT, nCorrect, nProposals, loss_x.item(), loss_y.item(), loss_conf.item(), loss.item()))

        return loss
