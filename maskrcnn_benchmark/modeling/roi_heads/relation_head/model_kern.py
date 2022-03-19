# Well, this file contains modules of GGNN_obj and GGNN_rel
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from maskrcnn_benchmark.config import cfg


class GGNNObj(nn.Module):
    """
    the context message passing module for the instance context
    """

    def __init__(self, num_obj_cls=151, time_step_num=3, hidden_dim=512, output_dim=512, use_prior_prob_knowledge=True,
                 prior_matrix=''):
        """

        :param num_obj_cls:
        :param time_step_num:
        :param hidden_dim:
        :param output_dim:
        :param use_prior_prob_knowledge: query from the statistics occurrence probability prior knowledge
        :param prior_matrix:
        """
        super(GGNNObj, self).__init__()
        self.num_obj_cls = num_obj_cls
        self.time_step_num = time_step_num
        self.output_dim = output_dim

        if use_prior_prob_knowledge:
            matrix_np = np.load(prior_matrix).astype(np.float32)
        else:
            matrix_np = np.ones((num_obj_cls, num_obj_cls)).astype(np.float32) / num_obj_cls

        # didn't finetuned during the training
        self.matrix = nn.Parameter(torch.from_numpy(matrix_np), requires_grad=False)
        # if you want to use multi gpu to run this model, then you need to use the following line code to replace the last line code.
        # And if you use this line code, the model will save prior matrix as parameters in saved models.
        # self.matrix = nn.Parameter(torch.from_numpy(matrix_np), requires_grad=False)

        # here we follow the paper "Gated graph sequence neural networks" to implement GGNN, so eq3 means equation 3 in this paper.
        self.fc_eq3_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq3_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq4_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq5_u = nn.Linear(hidden_dim, hidden_dim)

        self.fc_output = nn.Linear(2 * hidden_dim, output_dim)
        self.ReLU = nn.ReLU(True)
        self.fc_obj_cls = nn.Linear(self.num_obj_cls * output_dim, self.num_obj_cls)

    def forward(self, instance_feats):
        """

        :param instance_feats: batch concatenated instance features (num_instances, hidden_dim)
        :return:
        """
        # propogation process
        num_object = instance_feats.size()[0]
        # (num_instances, num_obj_cls, hidden_dim)
        hidden = instance_feats.repeat(1, self.num_obj_cls).view(num_object, self.num_obj_cls, -1)
        for t in range(self.time_step_num):
            # eq(2)
            # here we use some matrix operation skills
            hidden_sum = torch.sum(hidden, 0)
            av = torch.cat(
                [torch.cat([self.matrix.transpose(0, 1) @ (hidden_sum - hidden_i) for hidden_i in hidden], 0),
                 torch.cat([self.matrix @ (hidden_sum - hidden_i) for hidden_i in hidden], 0)], 1)

            # eq(3)
            hidden = hidden.view(num_object * self.num_obj_cls, -1)
            zv = torch.sigmoid(self.fc_eq3_w(av) + self.fc_eq3_u(hidden))

            # eq(4)
            rv = torch.sigmoid(self.fc_eq4_w(av) + self.fc_eq3_u(hidden))

            # eq(5)
            hv = torch.tanh(self.fc_eq5_w(av) + self.fc_eq5_u(rv * hidden))

            hidden = (1 - zv) * hidden + zv * hv
            hidden = hidden.view(num_object, self.num_obj_cls, -1)

        output = torch.cat((hidden.view(num_object * self.num_obj_cls, -1),
                            instance_feats.repeat(1, self.num_obj_cls).view(num_object * self.num_obj_cls, -1)), 1)
        output = self.fc_output(output)
        output = self.ReLU(output)
        obj_dists = self.fc_obj_cls(output.view(-1, self.num_obj_cls * self.output_dim))
        return obj_dists


class GGNNRel(nn.Module):
    def __init__(self, num_rel_cls=51, time_step_num=3, hidden_dim=512, output_dim=512, use_knowledge=True,
                 prior_matrix=''):
        super(GGNNRel, self).__init__()
        self.num_rel_cls = num_rel_cls
        self.time_step_num = time_step_num
        self.matrix = np.load(prior_matrix).astype(np.float32)
        self.use_knowledge = use_knowledge
        self.avg_graph_sum = cfg.MODEL.ROI_RELATION_HEAD.KERN_MODULE.AVERAGE_GRAPH_SUMMARY

        self.fc_eq3_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq3_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq4_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq5_u = nn.Linear(hidden_dim, hidden_dim)

        self.fc_output = nn.Linear(2 * hidden_dim, output_dim)
        self.ReLU = nn.ReLU(True)
        if not self.avg_graph_sum:
            self.fc_output_2 = nn.Linear((self.num_rel_cls + 2) * output_dim, output_dim)

    def forward(self, rel_inds, sub_obj_preds, input_ggnn):

        (input_rel_num, node_num, _) = input_ggnn.size()
        assert input_rel_num == len(rel_inds)
        batch_in_matrix_sub = np.zeros((input_rel_num, 2, self.num_rel_cls), dtype=np.float32)

        if self.use_knowledge:  # construct adjacency matrix depending on the predicted labels of subject and object.
            for index, rel in enumerate(rel_inds):
                batch_in_matrix_sub[index][0] = \
                    self.matrix[sub_obj_preds[index, 0].cpu().data, sub_obj_preds[index, 1].cpu().data]
                batch_in_matrix_sub[index][1] = batch_in_matrix_sub[index][0]
        else:
            for index, rel in enumerate(rel_inds):
                batch_in_matrix_sub[index][0] = 1.0 / float(self.num_rel_cls)
                batch_in_matrix_sub[index][1] = batch_in_matrix_sub[index][0]
        batch_in_matrix_sub_gpu = Variable(torch.from_numpy(batch_in_matrix_sub), requires_grad=False).cuda()
        del batch_in_matrix_sub

        hidden = input_ggnn
        hidden_save = []
        for t in range(self.time_step_num):
            # eq(2)
            # becase in this case, A^(out) == A^(in), so we use function "repeat"
            # What is A^(out) and A^(in)? Please refer to paper "Gated graph sequence neural networks"
            # aggregation between the object features and predicates features
            av = torch.cat((torch.bmm(batch_in_matrix_sub_gpu, hidden[:, 2:]),  # predicate features to object features
                            # rel_num x (2 x rel_cate_num @ rel_cate_num x 2) = rel_num x 2 x dim
                            torch.bmm(batch_in_matrix_sub_gpu.transpose(1, 2), hidden[:, :2])),
                           # object features to predicate features
                           # rel_num x (rel_cate_num x 2 @ 2 x dim) = rel_num x rel_cate_num x dim
                           dim=1
                           ).repeat(1, 1, 2)  # rel_num x (rel_cate_num + 2) x (dim x 2) # 为什么要 repeat ???
            # update the node according to the aggregation features
            av = av.view(input_rel_num * node_num, -1)  # flatten the batch wise dimension for parallel
            flatten_hidden = hidden.view(input_rel_num * node_num, -1)
            # eq(3)
            zv = torch.sigmoid(self.fc_eq3_w(av) + self.fc_eq3_u(flatten_hidden))
            # eq(4)
            rv = torch.sigmoid(self.fc_eq4_w(av) + self.fc_eq3_u(flatten_hidden))
            # eq(5)
            hv = torch.tanh(self.fc_eq5_w(av) + self.fc_eq5_u(rv * flatten_hidden))
            flatten_hidden = (1 - zv) * flatten_hidden + zv * hv
            hidden = flatten_hidden.view(input_rel_num, node_num, -1)
            hidden_save.append(hidden[:])

        if not self.avg_graph_sum:
            output = torch.cat((flatten_hidden, input_ggnn.view(input_rel_num * node_num, -1)), 1)
            output = self.fc_output(output)
            output = self.ReLU(output)
            output = self.fc_output_2(output.view(input_rel_num, -1))
            return output
        else:
            graph_out = torch.cat((flatten_hidden.reshape(input_rel_num, node_num, -1), input_ggnn), 2)
            graph_out = graph_out.mean(dim=1)
            return self.fc_output(graph_out)


class GGNNObjReason(nn.Module):
    """
    Module for object classification
    """

    def __init__(self, mode='sgdet', num_obj_cls=151, obj_dim=4096,
                 time_step_num=3, hidden_dim=512, output_dim=512,
                 use_knowledge=True, knowledge_matrix=''):
        super(GGNNObjReason, self).__init__()
        self.mode = mode
        self.num_obj_cls = num_obj_cls
        self.obj_proj = nn.Linear(obj_dim, hidden_dim)
        self.ggnn_obj = GGNNObj(num_obj_cls=num_obj_cls, time_step_num=time_step_num, hidden_dim=hidden_dim,
                                output_dim=output_dim, use_knowledge=use_knowledge, prior_matrix=knowledge_matrix)

    def forward(self, im_inds, obj_fmaps, obj_labels):
        """
        Reason object classes using knowledge of object cooccurrence
        """

        if self.mode == 'predcls':
            # in task 'predcls', there is no need to run GGNN_obj
            obj_dists = Variable(to_onehot(obj_labels.data, self.num_obj_cls))
            return obj_dists
        else:
            input_ggnn = self.obj_proj(obj_fmaps)

            lengths = []
            for i, s, e in enumerate_by_image(im_inds.data):
                lengths.append(e - s)
            obj_cum_add = np.cumsum([0] + lengths)
            obj_dists = torch.cat(
                [self.ggnn_obj(input_ggnn[obj_cum_add[i]: obj_cum_add[i + 1]]) for i in range(len(lengths))], 0)
            return obj_dists


class GGNNRelReason(nn.Module):
    """
    Module for relationship classification.
    """

    def __init__(self, num_obj_cls=151, num_rel_cls=51, inst_feat_dim=4096, rel_feat_dim=4096,
                 time_step_num=3, hidden_dim=512, output_dim=512,
                 use_knowledge=True, knowledge_matrix=''):
        super(GGNNRelReason, self).__init__()
        # mode
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        self.num_obj_cls = num_obj_cls
        self.num_rel_cls = num_rel_cls
        self.obj_dim = inst_feat_dim
        self.rel_dim = rel_feat_dim

        self.instance_fc = nn.Linear(self.obj_dim, hidden_dim)
        self.rel_union_feat_fc = nn.Linear(self.rel_dim, hidden_dim)

        self.ggnn_rel = GGNNRel(num_rel_cls=num_rel_cls, time_step_num=time_step_num, hidden_dim=hidden_dim,
                                output_dim=output_dim, use_knowledge=use_knowledge, prior_matrix=knowledge_matrix)

    def forward(self, inst_feats, union_feats, sub_obj_preds, rel_pair_idxs):
        """
        Reason relationship classes using knowledge of object and relationship coccurrence.
        all features vectors are batch concatenated
        :param inst_feats: (num_instances, hidden_dim)
        :param rel_pair_idxs: num_rel, 2
        :param union_feats: num_rel, pooling_dim
        :param inst_pred_labels: instance prediction labels, pass GT while training
        :return:
        """

        batched_rel_pair_idx = []
        start = 0
        for idx in range(len(rel_pair_idxs)):
            batched_rel_pair_idx.append(rel_pair_idxs[idx] + start)
            start += torch.max(rel_pair_idxs[idx])
        batched_rel_pair_idx = torch.cat(batched_rel_pair_idx)

        inst_feats = self.instance_fc(inst_feats)
        union_feats = self.rel_union_feat_fc(union_feats)
        # (num_rel, num_rel_cls + 2, hidden_dim )
        gnn_input = torch.stack([torch.cat([inst_feats[rel_ind[0]].unsqueeze(0),  # 1, hidden_dim
                                            inst_feats[rel_ind[1]].unsqueeze(0),  # 1, hidden_dim
                                            union_feats[index].repeat(self.num_rel_cls, 1)], 0)  # 51, hidden_dim
                                 for index, rel_ind in enumerate(batched_rel_pair_idx)])
        rel_dists = self.ggnn_rel(batched_rel_pair_idx, sub_obj_preds, gnn_input)

        return rel_dists


def to_onehot(vec, num_classes, fill=1000):
    """
    Creates a [size, num_classes] torch FloatTensor where
    one_hot[i, vec[i]] = fill

    :param vec: 1d torch tensor
    :param num_classes: int
    :param fill: value that we want + and - things to be.
    :return:
    """
    onehot_result = vec.new(vec.size(0), num_classes).float().fill_(-fill)
    arange_inds = vec.new(vec.size(0)).long()
    torch.arange(0, vec.size(0), out=arange_inds)

    onehot_result.view(-1)[vec + num_classes * arange_inds] = fill
    return onehot_result


import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.utils import cat
from .utils_motifs import obj_edge_vectors, encode_box_info, to_onehot


# simlar to the Motifs LSTM pipeline, but we just use to generate the object pair features
# such as labels space embedding

class InstanceFeaturesAugments(nn.Module):
    """
    extract the pairwise features from the object pairs and union features.
    most pipeline keep same with the motifs instead the lstm massage passing process
    """

    def __init__(self, config, obj_classes, in_channels, use_obj_pairwise_feats):
        super(InstanceFeaturesAugments, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.num_obj_classes = len(obj_classes)

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        # word embedding
        # add language prior representation according to the prediction distribution
        # of objects
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.obj_dim = in_channels
        self.use_obj_pairwise_feats = use_obj_pairwise_feats
        self.obj_rep_out_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.rel_rep_out_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.word_embed_feats_on = config.MODEL.ROI_RELATION_HEAD.WORD_EMBEDDING_FEATURES
        if self.word_embed_feats_on:
            obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
            self.obj_embed_on_1stg_pred = nn.Embedding(self.num_obj_classes, self.embed_dim)
            self.obj_embed_on_2stg_pred = nn.Embedding(self.num_obj_classes, self.embed_dim)
            with torch.no_grad():
                self.obj_embed_on_1stg_pred.weight.copy_(obj_embed_vecs, non_blocking=True)
                self.obj_embed_on_2stg_pred.weight.copy_(obj_embed_vecs, non_blocking=True)
        else:
            self.embed_dim = 0

        self.effect_analysis = False
        # position embedding
        # encode the geometry information of bbox in relationships
        self.geometry_feat_dim = 128
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, self.geometry_feat_dim), nn.ReLU(inplace=True),
        ])

        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.obj_hidden_linear = nn.Linear(self.obj_dim + self.embed_dim + self.geometry_feat_dim,
                                           self.obj_rep_out_dim)
        if self.use_obj_pairwise_feats:
            self.edges_hidden_linear = nn.Linear(self.obj_rep_out_dim + self.obj_dim + self.embed_dim,
                                                 self.rel_rep_out_dim)

        # untreated average features
        self.average_ratio = 0.0005

    def object_feature_refine(self, obj_feats, proposals, obj_labels=None, ctx_average=False):
        """
        Object feature refinement by embedding representation and redo classification on new representation.
        all vectors from each images of batch are cat together
        :param obj_feats: [num_obj, ROI feat dim + object embedding0 dim + geometry_feat_dim]
        :param obj_labels: [num_obj] the GT labels of the image
        :param proposals: BoxList for objects
        :param boxes_per_cls: regressed boxes for each categories

        :return: obj_pred_logits: [num_obj, #classes] new probability distribution.
                 obj_preds: [num_obj, ] argmax of that distribution.
                 augmented_obj_features: [num_obj, #feats] For later!
        """

        # fuse the ebedding features
        augmented_obj_features = self.obj_hidden_linear(obj_feats)  # map to hidden_dim

        # untreated decoder input
        batch_size = augmented_obj_features.shape[0]

        # todo causal module
        # if (not self.training) and self.effect_analysis and ctx_average:
        #     augmented_obj_features = self.untreated_obj_pairwise_dowdim_feat.view(1, -1).expand(batch_size, -1)
        #
        # if self.training and self.effect_analysis:
        #     self.untreated_obj_pairwise_dowdim_feat = self.moving_average(self.untreated_obj_pairwise_dowdim_feat,
        #                                                                   augmented_obj_features)

        # todo reclassify on the fused object features
        # Decode in order
        if self.mode != 'predcls':
            # todo: currently no redo classification on embedding representation,
            #       we just use the first stage object prediction
            obj_pred_labels = cat([each_prop.get_field("pred_labels") for each_prop in proposals], dim=0)
            obj_pred_logits = cat([each_prop.get_field("predict_logits") for each_prop in proposals], dim=0)

        else:
            assert obj_labels is not None
            obj_pred_labels = obj_labels
            obj_pred_logits = to_onehot(obj_pred_labels, self.num_obj_classes)

        return augmented_obj_features, obj_pred_labels, obj_pred_logits

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def forward(self, inst_roi_feats, inst_proposals, logger=None, all_average=False, ctx_average=False):
        """

        :param inst_roi_feats: instance ROI features, list(Tensor)
        :param inst_proposals: instance proposals, list(BoxList())
        :param rel_pair_idxs:
        :param logger:
        :param all_average:
        :param ctx_average:
        :return:
            obj_pred_logits obj_pred_labels 2nd time instance classification results
            obj_representation4rel, the objects features ready for the represent the relationship
        """
        # using label or logits do the label space embeddings
        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            obj_labels = cat([proposal.get_field("labels") for proposal in inst_proposals], dim=0)
        else:
            obj_labels = None

        if self.word_embed_feats_on:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL and self.mode == "precls":
                obj_embed_by_pred_dist = self.obj_embed_on_1stg_pred(obj_labels.long())
            else:
                obj_logits = cat([proposal.get_field("predict_logits") for proposal in inst_proposals], dim=0).detach()
                obj_embed_by_pred_dist = F.softmax(obj_logits, dim=1) @ self.obj_embed_on_1stg_pred.weight

        # box positive geometry embedding
        assert inst_proposals[0].mode == 'xyxy'
        pos_embed = self.pos_embed(encode_box_info(inst_proposals))

        batch_size = inst_roi_feats.shape[0]
        if all_average and self.effect_analysis and (not self.training):
            obj_pre_rep = self.untreated_obj_init_feat.view(1, -1).expand(batch_size, -1)
        else:
            if self.word_embed_feats_on:
                obj_pre_rep = cat((inst_roi_feats, obj_embed_by_pred_dist, pos_embed), -1)
            else:
                obj_pre_rep = cat((inst_roi_feats, pos_embed), -1)

        # object level contextual feature
        augment_obj_feat, obj_pred_labels, obj_pred_logits = self.object_feature_refine(obj_pre_rep,
                                                                                        inst_proposals,
                                                                                        obj_labels)
        obj_representation4rel = None
        if self.use_obj_pairwise_feats:
            # object labels space embedding from the prediction labels
            if self.word_embed_feats_on:
                obj_embed_by_pred_labels = self.obj_embed_on_2stg_pred(obj_pred_labels.long())

            # average action in test phrase for causal effect analysis
            if (all_average or ctx_average) and self.effect_analysis and (not self.training):
                # average the embedding and initial ROI features
                obj_representation4rel = cat(
                    (self.untreated_obj_pairwised_feat.view(1, -1).expand(batch_size, -1), augment_obj_feat),
                    dim=-1)
            else:
                if self.word_embed_feats_on:
                    obj_representation4rel = cat((obj_embed_by_pred_labels, inst_roi_feats, augment_obj_feat), -1)
                else:
                    obj_representation4rel = cat((inst_roi_feats, augment_obj_feat), -1)

            # mapping to hidden
            obj_representation4rel = self.edges_hidden_linear(obj_representation4rel)

        return obj_representation4rel, augment_obj_feat, obj_pred_logits, obj_pred_labels,
