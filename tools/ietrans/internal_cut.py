import json
import pickle
import sys
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
import IPython

topk_percent = float(sys.argv[1])
path = "em_E.pk"

vocab = json.load(open("VG-SGG-dicts-with-attri.json", "r"))
idx2lb = {int(k): v for k, v in vocab["idx_to_label"].items()}
lb2idx = {k: int(v) for k, v in vocab["label_to_idx"].items()}
idx2pred = {int(k)-1: v for k, v in vocab["idx_to_predicate"].items()}
pred2idx = {k: int(v)-1 for k, v in vocab["predicate_to_idx"].items()}

l = pickle.load(open(path, "rb"))

rel_dic = {}
rel_cnt_dic = {}
all_triplet_idxs = {}
all_triplet_subs = []
all_triplet_objs = []
all_triplet_rels = []
all_triplet_logits = []
n = 0
for i, data in enumerate(l):
    labels = data["labels"]
    logits = data["logits"][:, 1:]
    relation_tuple = deepcopy(data["relations"])
    sub_idxs, obj_idxs, rels = relation_tuple[:, 0], relation_tuple[:, 1], relation_tuple[:, 2]
    sub_lbs, obj_lbs = labels[sub_idxs], labels[obj_idxs]
    # [[sub_lb1, obj_lb1], [sub_lb2, obj_lb2]......]
    pairs = np.stack([sub_lbs, obj_lbs], 1).tolist()
    pairs = [(idx2lb[p[0]], idx2lb[p[1]]) for p in pairs]
    # behave as indexes, so -=1
    rels -= 1

    # fill in rel_dic
    # rel_dic: {rel_i: {pair_j: distribution} }
    for j, (pair, r, logit) in enumerate(zip(pairs, rels, logits)):
        r_name = idx2pred[int(r)]

        if r_name not in rel_cnt_dic:
            rel_cnt_dic[r_name] = {}
        if pair not in rel_cnt_dic[r_name]:
            rel_cnt_dic[r_name][pair] = 0
        rel_cnt_dic[r_name][pair] += 1

        logit = torch.softmax(torch.from_numpy(logit), 0)
        if r_name not in rel_dic:
            rel_dic[r_name] = {}
        if pair not in rel_dic[r_name]:
            rel_dic[r_name][pair] = 0
        rel_dic[r_name][pair] += logit

        all_triplet_rels.append(r)
        all_triplet_subs.append(lb2idx[pair[0]])
        all_triplet_objs.append(lb2idx[pair[1]])
        all_triplet_logits.append(logit)
        all_triplet_idxs[n] = (i, j)
        n += 1

all_triplet_rels = np.asarray(all_triplet_rels)
all_triplet_subs = np.asarray(all_triplet_subs)
all_triplet_objs = np.asarray(all_triplet_objs)
all_triplet_logits = torch.stack(all_triplet_logits, 0)
print(len(all_triplet_rels), len(all_triplet_subs), len(all_triplet_objs), all_triplet_logits.size())
assert len(all_triplet_rels) == len(all_triplet_subs) == len(all_triplet_objs) == len(all_triplet_logits)
assert n == len(all_triplet_rels)
assert len(all_triplet_idxs) == len(all_triplet_rels)
all_changes = np.zeros_like(all_triplet_rels, dtype=np.float)

def vis_triplet(triplet):
    logit = rel_dic[triplet[0]][(triplet[1], triplet[2])]
    scores, idxs = logit.sort(descending=True)
    prds = [idx2pred.get(int(i), "none") for i in idxs]
    return list(zip(prds, scores))


def find_wrong_most_triplet(r):
    cnt = sum(rel_cnt_dic[r].values())
    # {pair_j: distribution}
    pair_dic = deepcopy(rel_dic[r])
    pair_dic = {pair: logit.softmax(0) for pair, logit in pair_dic.items()}
    pair_dic = { pair: logit[ int( pred2idx[r] ) ] for pair, logit in pair_dic.items() }
    ret = sorted(pair_dic.items(), key=lambda k: k[-1])
    ret = [(t[0], t[1], rel_cnt_dic[r][t[0]], cnt) for t in ret]
    return ret


def collect_all_parent_data(query_parents, query_pair, son_rel_idx):
    """
    Args:
        query_parents:
        query_pair:
        son_rel_idx:

    Returns:
        [(i, j), score_of_son_rel]
    """
    sub_lb, obj_lb = lb2idx[query_pair[0]], lb2idx[query_pair[1]]
    pair_flag = (all_triplet_subs==sub_lb) & (all_triplet_objs==obj_lb)
    rel_flag = np.zeros_like(pair_flag) != 0
    for p in query_parents:
        qp_idx = pred2idx[p]
        rel_flag |= (all_triplet_rels == qp_idx)
    flag = rel_flag & pair_flag
    logits = all_triplet_logits[flag, son_rel_idx]
    return np.where(flag)[0], logits


# construct importance dic, which is the attractor factors in the paper
importance_dic = {}
for r, pair_cnt_dic in rel_cnt_dic.items():
    for pair in pair_cnt_dic:
        cnt = pair_cnt_dic[pair]
        triplet = (r, *pair)
        importance_dic[triplet] = cnt/sum(pair_cnt_dic.values())

# list all triplets
all_triplets = []
for r, pair_cnt_dic in rel_cnt_dic.items():
    for pair in pair_cnt_dic:
        all_triplets.append((r, *pair))

# use vis_triplet
for triplet in tqdm(all_triplets):
    # IPython.embed()
    # triplet: (r, sub, obj)
    r = triplet[0]
    # prds: [(rel, score)]
    prds = vis_triplet(triplet)
    # find parents
    parents = [p[0] for p in prds]
    parents = parents[: parents.index(r)]
    # filter parents:
    # if current triplet is more important for node, the node is a son
    parents = [ p for p in parents if importance_dic.get((p, triplet[1], triplet[2]), 0) < importance_dic[triplet] ]

    # get data from parents
    # collect all parent data
    idxs, logits = collect_all_parent_data(parents, (triplet[1], triplet[2]), pred2idx[r])
    _, sorted_idxs = logits.sort(descending=True)
    idxs = idxs[sorted_idxs.numpy()]
    idxs = idxs[ 0: int(len(idxs)*topk_percent) ]
    # resolve conflicts
    mask = all_changes[idxs] < importance_dic[triplet]
    idxs = idxs[mask]
    for index in idxs:
        i, j = all_triplet_idxs[index]
        l[i]["relations"][j, 2] = pred2idx[r] + 1
    # modify relations
    all_changes[idxs] = importance_dic[triplet]
pickle.dump(l, open("em_E.pk_topk_"+str(round(topk_percent, 3)), "wb"))


