import os
import sys
import torch
import h5py
import json
from PIL import Image
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random

from itertools import product
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from .visual_genome import load_info, load_image_filenames, correct_img_info, get_VG_statistics
import pickle
BOX_SCALE = 1024  # Scale at which we have the boxes


class ExTransDataset(torch.utils.data.Dataset):

    def __init__(self, split, img_dir, roidb_file, dict_file, image_file, transforms=None,
                 filter_empty_rels=True, num_im=-1, num_val_im=5000,
                 filter_duplicate_rels=True, filter_non_overlap=True, flip_aug=False, custom_eval=False,
                 custom_path='', distant_supervsion_file=None, specified_data_file=None, custom_bbox_path=''):
        """
        The dataset to conduct external transfer
        Parameters:
            split: Must be train, test, or val
            img_dir: folder containing all vg images
            roidb_file:  HDF5 containing the GT boxes, classes, and relationships
            dict_file: JSON Contains mapping of classes/relationships to words
            image_file: HDF5 containing image filenames
            filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
            filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
            num_im: Number of images in the entire dataset. -1 for all images.
            num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
            specified_data_file: pickle file constains training data
        """
        # for debug
        # num_im = 10000
        # num_val_im = 4
        assert split in {'train'}
        assert flip_aug is False
        self.flip_aug = flip_aug
        self.split = split
        self.img_dir = img_dir
        self.dict_file = dict_file
        self.roidb_file = roidb_file
        self.image_file = image_file
        self.filter_non_overlap = filter_non_overlap and self.split == 'train'
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.transforms = transforms
        self.distant_supervision_bank = json.load(open(distant_supervsion_file, "r"))

        self.ind_to_classes, self.ind_to_predicates, self.ind_to_attributes = load_info(
            dict_file)  # contiguous 151, 51 containing __background__
        self.num_rel_classes = len(self.ind_to_predicates)
        self.categories = {i: self.ind_to_classes[i] for i in range(len(self.ind_to_classes))}

        self.custom_eval = custom_eval
        if self.custom_eval:
            self.get_custom_imgs(custom_path)
        else:
            assert filter_non_overlap is False
            assert filter_empty_rels is True
            self.split_mask, self.gt_boxes, self.gt_classes, self.gt_attributes, self.relationships = load_graphs(
                self.roidb_file, self.split, num_im, num_val_im=num_val_im,
                filter_empty_rels=filter_empty_rels,
                filter_non_overlap=self.filter_non_overlap,
                dskb=self.distant_supervision_bank
            )
            self.relationships = None
            self.filenames, self.img_info = load_image_filenames(img_dir, image_file)  # length equals to split_mask
            self.filenames = [self.filenames[i] for i in np.where(self.split_mask)[0]]
            self.img_info = [self.img_info[i] for i in np.where(self.split_mask)[0]]
        self.data = pickle.load(open(specified_data_file, "rb"))

        # check
        assert len(self.data) == len(self.img_info), "{}, {}".format(len(self.data), len(self.img_info))
        mask = [k is not None for k in self.data]
        self.data = [d for m, d in zip(mask, self.data) if m]
        self.img_info = [d for m, d in zip(mask, self.img_info) if m]
        self.filenames = [d for m, d in zip(mask, self.filenames) if m]
        self.gt_attributes = [d for m, d in zip(mask, self.gt_attributes) if m]
        for a, b in zip(self.data, self.filenames):
            assert a['img_path'] == b


    def __getitem__(self, index):
        img = Image.open(self.filenames[index]).convert("RGB")
        target = self.get_groundtruth(index)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        target.add_field("cur_data", self.data[index])
        return img, target, index

    # def get_statistics(self):
    #     result = {
    #         'fg_matrix': None,
    #         'pred_dist': None,
    #         'obj_classes': self.ind_to_classes,
    #         'rel_classes': self.ind_to_predicates,
    #         'att_classes': self.ind_to_attributes,
    #     }
    #     return result

    def get_statistics(self, no_matrix=False):
        if no_matrix:
            return {
                'fg_matrix': None,
                'pred_dist': None,
                'obj_classes': self.ind_to_classes,
                'rel_classes': self.ind_to_predicates,
                'att_classes': self.ind_to_attributes,
            }
        fg_matrix, bg_matrix = get_VG_statistics(img_dir=self.img_dir, roidb_file=self.roidb_file,
                                                 dict_file=self.dict_file,
                                                 image_file=self.image_file, must_overlap=True)
        eps = 1e-3
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
            'att_classes': self.ind_to_attributes,
        }
        return result

    def get_custom_imgs(self, path):
        self.custom_files = []
        self.img_info = []
        for file_name in os.listdir(path):
            self.custom_files.append(os.path.join(path, file_name))
            img = Image.open(os.path.join(path, file_name)).convert("RGB")
            self.img_info.append({'width': int(img.width), 'height': int(img.height)})

    def get_img_info(self, index):
        # WARNING: original image_file.json has several pictures with false image size
        # use correct function to check the validity before training
        # it will take a while, you only need to do it once

        # correct_img_info(self.img_dir, self.image_file)
        return self.img_info[index]

    def get_groundtruth(self, index, flip_img=False):
        cur_data = self.data[index]

        img_info = self.get_img_info(index)
        w, h = img_info['width'], img_info['height']

        box = torch.from_numpy(cur_data['boxes']).reshape(-1, 4)  # guard against no boxes
        target = BoxList(box, (w, h), 'xyxy')  # xyxy
        target.add_field("labels", torch.from_numpy(cur_data['labels']))
        target.add_field("attributes", torch.from_numpy(self.gt_attributes[index])[:len(box)])
        target.add_field("relation_pair_idxs", torch.from_numpy(cur_data['pairs']))

        # relation_map_tensor = torch.zeros((len(cur_data['pairs']), self.num_rel_classes), dtype=torch.float32)
        # relation_soft_labels = torch.zeros((len(cur_data['pairs']), self.num_rel_classes), dtype=torch.float32)
        # rel_logits = cur_data['rel_logits']
        # possible_rels = cur_data['possible_rels']
        # for i, (logit, ds_rels) in enumerate(zip(rel_logits, possible_rels)):
        #     if logit is None:
        #         logit = torch.tensor([1.])
        #     else:
        #         logit = (torch.from_numpy(logit).float().squeeze(0)).softmax(0)
        #     relation_soft_labels[i, ds_rels] = logit
        #     relation_map_tensor[i, ds_rels] = 1.
        # target.add_field("relation_labels", relation_map_tensor)
        # target.add_field("relation_soft_labels", relation_soft_labels)
        # target.add_field("relation_labels", relation_soft_labels)
        return target

    def __len__(self):
        if self.custom_eval:
            return len(self.custom_files)
        return len(self.filenames)


def load_graphs(roidb_file, split, num_im, num_val_im, filter_empty_rels, filter_non_overlap, dskb=None):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    Parameters:
        roidb_file: HDF5
        split: (train, val, or test)
        num_im: Number of images we want
        num_val_im: Number of validation images
        filter_empty_rels: (will be filtered otherwise.)
        filter_non_overlap: If training, filter images that dont overlap.
    Return:
        image_index: numpy array corresponding to the index of images we're using
        boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
        gt_classes: List where each element is a [num_gt] array of classes
        relationships: List where each element is a [num_r, 3] array of
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    roi_h5 = h5py.File(roidb_file, 'r')
    data_split = roi_h5['split'][:]
    split_dic = {'train': 0, 'val': 1, 'test': 2}
    split_flag = split_dic[split]
    # if split == 'val':
    #     split_mask = (data_split == 1) | (data_split == 2)
    split_mask = data_split == split_flag

    # Filter out images without bounding boxes
    split_mask &= roi_h5['img_to_first_box'][:] >= 0
    # split_mask &= (roi_h5['img_to_last_box'][:]-roi_h5['img_to_first_box'][:]) > 1

    image_index = np.where(split_mask)[0]
    if num_im > -1:
        image_index = image_index[:num_im]

    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True

    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    if 'attributes' not in roi_h5:
        all_attributes = np.zeros((roi_h5["boxes_1024"].shape[0], 1))
    else:
        all_attributes = roi_h5['attributes'][:, :]
    all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # cx,cy,w,h
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    # assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    boxes = []
    gt_classes = []
    gt_attributes = []
    # relationships = []
    for i in range(len(image_index)):
        i_obj_start = im_to_first_box[i]
        i_obj_end = im_to_last_box[i]

        boxes_i = all_boxes[i_obj_start : i_obj_end + 1, :]
        gt_classes_i = all_labels[i_obj_start : i_obj_end + 1]
        gt_attributes_i = all_attributes[i_obj_start : i_obj_end + 1, :]

        if filter_empty_rels:
            assert dskb is not None
            n = 0
            for sub_idx, obj_idx in product(range(len(gt_classes_i)), repeat=2):
                if sub_idx != obj_idx:
                    n += (dskb.get(str(gt_classes_i[sub_idx].item()) + '_' + str(gt_classes_i[obj_idx].item()), None) is not None)
            if n == 0:
                split_mask[image_index[i]] = 0
                continue

        if filter_non_overlap:
            assert split == 'train'
            # construct BoxList object to apply boxlist_iou method
            # give a useless (height=0, width=0)
            boxes_i_obj = BoxList(boxes_i, (1000, 1000), 'xyxy')
            inters = boxlist_iou(boxes_i_obj, boxes_i_obj)
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        gt_attributes.append(gt_attributes_i)
        # relationships.append(rels)

    return split_mask, boxes, gt_classes, gt_attributes, None
