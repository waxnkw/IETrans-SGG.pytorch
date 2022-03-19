# modified from https://github.com/rowanz/neural-motifs
from maskrcnn_benchmark.modeling import registry
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn import functional as F
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.modeling.make_layers import make_fc
from .utils_motifs import obj_edge_vectors, encode_box_info, to_onehot
import json
import numpy as np


class KBBias(nn.Module):
    """
    The goal of this is to provide a simplified way of computing
    P(predicate | obj1, obj2, img).
    """

    def __init__(self, cfg):
        super(KBBias, self).__init__()
        self.num_rels = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.kb = json.load(open("datasets/vg/20/VGKB.json", "r"))

    def index_with_labels(self, labels):
        """
        :param labels: [batch_size, 2]
        :return:
        """
        hs = labels[:, 0].cpu().tolist()
        ts = labels[:, 1].cpu().tolist()
        get_rels = lambda k: self.kb.get(k, [0])
        keys = [get_rels(str(i)+"_"+str(j)) for i, j in zip(hs, ts)]
        keys = [np.random.choice(k) for k in keys]
        ret = torch.zeros((labels.size(0), self.num_rels), device=labels.device, dtype=torch.float32)
        ret[torch.arange(0, ret.size(0)), keys] = 1.
        return ret

    def forward(self, labels):
        # implement through index_with_labels
        return self.index_with_labels(labels)

class UnionPair(nn.Module):
    def __init__(self, config, num_obj, num_rel, in_channels, hidden_dim=512, num_iter=3):
        super(UnionPair, self).__init__()
        self.cfg = config
        self.num_obj = num_obj
        self.num_rel = num_rel
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.hidden_dim = hidden_dim
        self.num_iter = num_iter
        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'
        self.rel_fc = make_fc(self.pooling_dim, self.num_rel)
        # if not self.use_faster_rcnn:
        self.obj_fc = make_fc(self.pooling_dim, self.num_obj)

    def forward(self, x, proposals, union_features,  logger=None):
        if self.mode == 'predcls':
            obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
            obj_dists = to_onehot(obj_labels, self.num_obj)
        else:
            obj_dists = self.obj_fc(x)

        rel_dists = self.rel_fc(union_features)

        return obj_dists, rel_dists
