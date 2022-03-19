import json
import pickle
import sys
import numpy as np
import torch
ratio = float(sys.argv[1])
print(ratio)
score = json.load(open("score.json", "r"))
em = pickle.load(open("raw_em_E.pk", "rb"))
thres = score[int(ratio * len(score))-1]

n = 0
rst = []
for d in em:
    pairs = d['pairs']
    rel_logits = d['rel_logits']
    possible_rels = d['possible_rels']
    rst_pairs = []
    rst_rel_logits = []
    rst_possible_rels = []
    for i, (logit, rels) in enumerate(zip(rel_logits, possible_rels)):
        s = torch.tensor(logit).softmax(0)[-1]
        if s > thres:
            # do not transfer
            continue
        else:
            # transfer
            rst_pairs.append(pairs[i])
            rst_rel_logits.append(logit[0:-1])
            rst_possible_rels.append(rels[:-1])
    d['pairs'] = np.asarray(rst_pairs)
    d['rel_logits'] = rst_rel_logits
    d['possible_rels'] = rst_possible_rels
    if rst_rel_logits == []:
        n += 1
        rst.append(None)
    else:
        rst.append(d)
# pickle.dump(rst, open("em_E.pk"+str(round(ratio, 2)), "wb"))
pickle.dump(rst, open("em_E.pk", "wb"))
print(n)