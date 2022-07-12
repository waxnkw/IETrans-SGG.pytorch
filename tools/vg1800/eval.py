import json
import torch
from tqdm import tqdm
import sys
# n = int(sys.argv[1])

dic = torch.load("eval_results.pytorch")
# dic = torch.load("rst_debug.pytorch")

gts = dic["groundtruths"]
preds = dic["predictions"]

# create rel stat
rel_stat = {}
for gt in tqdm(gts):
    rels = gt.get_field("relation_tuple")[:, 2]
    for r in rels:
        rel_stat[int(r)] = rel_stat.get(int(r), 0) + 1
print(len(rel_stat))
assert len(rel_stat) == 1807

obj_stat = {}
for gt in tqdm(gts):
    objs = gt.get_field("labels")
    for lb in objs:
        obj_stat[int(lb)] = obj_stat.get(int(lb), 0) + 1
print(len(obj_stat))


def eval(gts, preds, n):
    tp_dic = {i: 0 for i in range(1, len(rel_stat)+1)}
    # calculate acc
    for gt, pre in tqdm(zip(gts, preds)):
        rels_tuples = gt.get_field("relation_tuple")
        gt_pairs = rels_tuples[:, 0:2]
        gt_rels = rels_tuples[:, 2]
        pair_idxs = pre.get_field("rel_pair_idxs")
        rel_scores = pre.get_field("pred_rel_scores")
        match_idxs = (gt_pairs[..., None] == pair_idxs.T[None, ...]).all(1).nonzero(as_tuple=False)
        # assert match_idxs.size(0) == gt_rels.size(0), IPython.embed()
        gt_rels = gt_rels[match_idxs[:, 0]]
        pred_scores = rel_scores[match_idxs[:, 1]]
        topk_pred_rels = pred_scores[:, 1:].float().topk(k=n, dim=1)[1] + 1
        correct = (gt_rels[..., None] == topk_pred_rels).any(1)
        for r, c in zip(gt_rels, correct):
            if c:
                tp_dic[int(r)] += 1
    tp_acc_dic = {}
    for k, v in tp_dic.items():
        if v <= 0:
            continue
        tp_acc_dic[k] = round(v / rel_stat[k] * 100, 2)
    return tp_acc_dic, tp_dic


def eval_obj(gts, preds, n):
    tp_dic = {k:0 for k in obj_stat}
    # calculate acc
    for gt, pre in tqdm(zip(gts, preds)):
        gt_labels = gt.get_field("labels")
        pred_labels = pre.get_field("pred_labels")
        correct = gt_labels==pred_labels
        for r, c in zip(gt_labels, correct):
            if c:
                tp_dic[int(r)] += 1
    tp_acc_dic = {}
    for k, v in tp_dic.items():
        if v <= 0:
            continue
        tp_acc_dic[k] = round(v / obj_stat[k] * 100, 2)
    return tp_acc_dic, tp_dic


def eval_triplet(gts, preds, n=1):
    tp_dic = {i: 0 for i in range(1, len(rel_stat)+1)}
    non_zero = {}
    # calculate acc
    for gt, pre in tqdm(zip(gts, preds)):
        rels_tuples = gt.get_field("relation_tuple")
        gt_pairs = rels_tuples[:, 0:2]
        gt_rels = rels_tuples[:, 2]
        pair_idxs = pre.get_field("rel_pair_idxs")
        rel_scores = pre.get_field("pred_rel_scores")
        match_idxs = (gt_pairs[..., None] == pair_idxs.T[None, ...]).all(1).nonzero(as_tuple=False)
        # assert match_idxs.size(0) == gt_rels.size(0), IPython.embed()
        gt_rels = gt_rels[match_idxs[:, 0]]
        pred_scores = rel_scores[match_idxs[:, 1]]
        topk_pred_rels = pred_scores[:, 1:].float().topk(k=n, dim=1)[1] + 1
        correct = (gt_rels[..., None] == topk_pred_rels).any(1)

        # object pair correctness
        gt_pairs = gt_pairs[match_idxs[:, 0]]
        gt_head_lbs = gt.get_field("labels")[gt_pairs[:, 0]]
        gt_end_lbs = gt.get_field("labels")[gt_pairs[:, 1]]
        pred_pairs = pair_idxs[match_idxs[:, 1]]
        pred_head_lbs = pre.get_field("pred_labels")[pred_pairs[:, 0]]
        pred_end_lbs = pre.get_field("pred_labels")[pred_pairs[:, 1]]
        head_correct = gt_head_lbs == pred_head_lbs
        end_correct = gt_end_lbs == pred_end_lbs

        # correct
        correct = head_correct&end_correct&correct

        for r, c, h, e in zip(gt_rels, correct, gt_head_lbs, gt_end_lbs):
            if c:
                tp_dic[int(r)] += 1
                s = "_".join([str(int(h)), str(int(r)), str(int(e))])
                non_zero[s] = 0
    tp_acc_dic = {}
    for k, v in tp_dic.items():
        if v <= 0:
            continue
        tp_acc_dic[k] = round(v / rel_stat[k] * 100, 2)
    print("rel_nonzero:", len(non_zero))
    return tp_acc_dic, tp_dic


def analyze(tp_acc_dic, tp_count_dic, rel_stat=rel_stat):
    print("acc\tnonzero\tmean_acc")
    acc = sum(tp_count_dic.values())/sum(rel_stat.values())*100
    macc = sum(tp_acc_dic.values())/len(rel_stat)
    print("{}\t{}\t{}\t{}".format(
        round(sum(tp_count_dic.values()) / sum(rel_stat.values()) * 100, 2),
        len([v for v in tp_count_dic.values() if v > 0]),
        round(sum(tp_acc_dic.values()) / len(rel_stat), 2),
        2 / (1 / acc + 1 / macc),
    ), )

tp_acc_dic, tp_count_dic = eval_obj(gts, preds, n=1)
print("obj:")
analyze(tp_acc_dic, tp_count_dic, obj_stat)
print()
for n in [1, 5, 10]:

    # tp_acc_dic, tp_count_dic = eval_obj(gts, preds, n)
    # print("obj")
    # analyze(tp_acc_dic, tp_count_dic, obj_stat)
    # print()
    #
    tp_acc_dic, tp_count_dic = eval(gts, preds, n)
    print("top{} predicate:".format(n))
    analyze(tp_acc_dic, tp_count_dic)
    print()

    tp_acc_dic, tp_count_dic = eval_triplet(gts, preds, n)
    print("top{} triplet:".format(n))
    analyze(tp_acc_dic, tp_count_dic)
    print()
# print([(k, rel_stat[k], v) for k, v in sorted(tp_acc_dic.items(), key=lambda k: k[-1], reverse=True)])






