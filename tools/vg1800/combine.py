import json, pickle
import numpy as np
import sys, os
from tqdm import tqdm
# extra: ['img_path', 'boxes', 'labels', 'pairs', 'possible_rels', 'rel_logits']
# intra: ['width', 'height', 'img_path', 'boxes', 'labels', 'relations', 'possible_rels', 'logits']
import sys

exp_dir = os.environ.get("EXP")
code_dir = os.environ.get("SG")
model_name = sys.argv[1]
int_relabel_name = str(sys.argv[2])
extra_path = os.path.join(exp_dir, "1800/{}/predcls/lt/external/relabel".format(model_name), "em_E.pk")
intra_path = os.path.join(exp_dir, "1800/{}/predcls/lt/internal/relabel".format(model_name), "em_E.pk_topk_{}".format(int_relabel_name))
print(extra_path)
print(intra_path)
img_info_path = os.path.join(code_dir, "datasets/vg/image_data.json")
extra_data = pickle.load(open(extra_path, "rb"))
intra_data = pickle.load(open(intra_path, "rb"))
img_infos = json.load(open(img_info_path, "r"))
img_infos = {k["image_id"]: k for k in img_infos}

# filter_out relations in extra
freq_rels = [0, 914, 607, 657, 1772, 913, 895, 3, 631, 95, 1697, 1380, 1512, 664, 583, 73]
# freq_rels = {r: 0 for r in freq_rels}
freq_rels = np.array(freq_rels, dtype=np.int64)


def get_img_id(path):
    return int(path.split("/")[-1].replace(".jpg", ""))


def to_dic(data):
    dic = {}
    for d in data:
        if d is None:
            continue
        dic[get_img_id(d['img_path'])] = d
    return dic


def complete_img_info(data, img_infos):
    imid = get_img_id(data["img_path"])
    iminfo = img_infos[imid]
    data["width"] = iminfo["width"]
    data["height"] = iminfo["height"]


def filter_out_freq_rels(relations):
    rels = relations[:, 2]
    in_mask = (rels.reshape(-1, 1) == freq_rels.reshape(-1, 1).T).any(-1)
    return relations[~in_mask]


def complete_relatinos(data):
    assert "rel_logits" in data, data
    rel_logits = data["rel_logits"]
    possible_rels = data["possible_rels"]
    pairs = data["pairs"]
    rels = []
    for poss_rls, logits in zip(possible_rels, rel_logits):
        max_id = logits.argmax()
        rels.append(poss_rls[max_id])
    rels = np.array(rels, dtype=pairs.dtype).reshape(-1, 1)
    rels = np.concatenate([pairs, rels], 1)
    data["relations"] = filter_out_freq_rels(rels)
    del data["pairs"]
    return data


def ex_data_to_in_data(ex_data, img_infos):
    complete_img_info(ex_data, img_infos)
    complete_relatinos(ex_data)
    del ex_data["rel_logits"]
    return ex_data


def merge_ex_and_in(ex_data, in_data):
    assert ex_data["img_path"] == in_data["img_path"]
    ex_data = ex_data_to_in_data(ex_data, img_infos)
    ex_rels = ex_data["relations"]
    ex_pairs = ex_rels[:, :2]
    in_rels = in_data["relations"]
    in_pairs = in_rels[:, :2]
    intersect_rels = ex_pairs.reshape(ex_pairs.shape[0], ex_pairs.shape[1], 1) \
                     == in_pairs.reshape(in_pairs.shape[0], in_pairs.shape[1], 1).T
    intersect_rels = intersect_rels.all(-1).any(1)
    unintersect_rels = ~intersect_rels
    final_rels = np.concatenate([in_rels, ex_rels[unintersect_rels]], 0)
    in_data["relations"] = final_rels
    return in_data


print("intra", len(intra_data), sum([len(x["relations"]) for x in intra_data]))
n = 0
# to dic
# extra: ['img_path', 'boxes', 'labels', 'pairs', 'possible_rels', 'rel_logits']
# intra: ['width', 'height', 'img_path', 'boxes', 'labels', 'relations', 'possible_rels', 'logits']
# output: 'img_path', 'boxes', 'labels', 'possible_rels', 'width', 'height', 'relations'
extra_data = to_dic(extra_data)
intra_data = to_dic(intra_data)
to_save_data = {}
keys = [x for x in extra_data] + [x for x in intra_data]
keys = list(set(keys))
for i, img_id in tqdm(enumerate(keys), total=len(keys)):
    in_data = intra_data.get(img_id, None)
    ex_data = extra_data.get(img_id, None)

    if not in_data and ex_data:
        tmp = ex_data_to_in_data(ex_data, img_infos)
        if len(tmp["relations"]) == 0:
            continue
        to_save_data[img_id] = tmp

    if in_data and not ex_data:
        to_save_data[img_id] = in_data

    if in_data and ex_data:
        n += in_data['labels'].shape[0] > ex_data['labels'].shape[0]
        to_save_data[img_id] = merge_ex_and_in(ex_data, in_data)
to_save_data = list(to_save_data.values())

# simplify
for d in to_save_data:
    if "logits" in d:
        del d["logits"]
    if "rel_logits" in d:
        del d["rel_logits"]
    if "pairs" in d:
        del d["pairs"]
    if "possible_rels" in d:
        del d["possible_rels"]

pickle.dump(to_save_data, open("em_E.pk", "wb"))

# stat relations
print("saved data:", len(to_save_data), sum([len(x["relations"]) for x in to_save_data]))
print(n)