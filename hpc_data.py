'''
House data loaders
- cifar10h 
- cifar10 (test) + our augmentations
- cifar10 (test) + our augmentations + modified vanilla mixup (?)
- cifar10h + our augmentations (?)
Note: specify data loaders to have varied number of samples
(against baseline of just cifar10, test set to match cifar10h)
'''

# create custom dataloaders in pytorch help from: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
from cProfile import label
from shutil import unregister_archive_format
from turtle import home
from typing import List
import numpy as np
import torch
import os 
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.datasets as datasets
from torch.autograd import Variable
import json
import torch.nn.functional as F
import pickle
import random
from label_construction_utils import construct_elicited_soft_label, create_cifar10h_sim2, get_semantic_sim_matrix
from scipy.stats import entropy, beta


def create_smoothed_label(hard_label_class, num_classes=10, smoothing_factor=0.1): 
    '''
    Construct a smoothed label from a provided hard label class
    '''
    hard_label = np.zeros([num_classes])
    hard_label[hard_label_class] = 1.0 
    # smoothed_label = np.ones([self.num_classes]) * (ls_alpha/(self.num_classes - 1)) # apply smoother to all classes which are not the hard label class
    # smoothed_label[hard_label] = 1.0 * (1-ls_alpha)
    smoothed_label = hard_label * (1-smoothing_factor) + np.ones([num_classes]) * smoothing_factor
    return smoothed_label

def fit_beta_params(prob_data,ent_type="low"):
    adjusted_prob_data= [] # for numerical precision
    for x in prob_data: 
        if x == 1.0:
            if ent_type == "low": 
                adjusted_prob_data.append(x-0.05) # b/c beta was too strong
            else: adjusted_prob_data.append(x)
        elif x==0.0: adjusted_prob_data.append(x+0.00001)
        else: adjusted_prob_data.append(x) 
    return beta.fit(adjusted_prob_data, floc=0.0)[:2]

def get_n_poss_dist(n_poss_data, valid_max_poss=9): 
   # store counts 
   valid_n_poss = list(range(valid_max_poss+1))
   tot_instances = len(n_poss_data)
   return [n_poss_data.count(n_poss)/tot_instances for n_poss in valid_n_poss]


class CIFARMixHILL(Dataset):
    """
    Data loader for CIFAR-10 (test set), augmented with HILL re-labeled samples
    modified from example: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, human_soft_labels_file="./data/cifar10h-probs.npy", img_dir='~/data',
                    transform=None, label_transform=None, split_idx_pth="./data/", split="train",
                    hill_mix_pth="./data/human_relabelings.json", label_method="avg", use_human_soft=False,
                    no_aug_set=False, use_all_data=False, aug_size=-1, hill_mix_soft_label_pth="./data/mixup_soft_labels.json",
                    rem_hill_mix_soft=False, just_hill_mix_soft=False,
                    smoother_a_param=50, smoother_b_param=0.0001):
        """
        Args:
            human_soft_labels_file (string, optional): Path to file with human soft labels (cifar10h)
            img_dir (string, optional): Pth to directory with CIFAR-10 (test) images used by Peterson, et al.
            transform (callable, optional): Optional transform to be applied
                on image.
            label_transform: Optional transform over label space, e.g., smoothing.
            modified from example: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        """
         # shape = [num_data_points, num_classes], e.g. [10,000, 10]
        if use_human_soft:
            self.human_soft_labels = np.load(human_soft_labels_file)
            self.human_label_counts = np.load("./data/cifar10h-counts.npy")

        self.img_dir = img_dir
        if img_dir is None: 
            # re-download cifar10 test set
            download_orig_cifar10 = True
            img_dir = "~/data" # save to
        else: download_orig_cifar10 = False
        self.img_dataset = datasets.CIFAR10(root=img_dir, train=False, download=download_orig_cifar10,
                        transform=transform) 
        data_size = len(self.img_dataset.data)
        self.imgs = self.img_dataset.data

        # extract names of classes
        self.classes = self.get_classes()
        # map from names of classes to indices
        self.class2idx = {class_name: idx for idx,class_name in enumerate(self.classes)}
        self.idx2class = {idx: class_name for class_name,idx in self.class2idx.items()}
        self.num_classes = len(self.classes)

        if use_human_soft:
            # for now, try w/ our generative model!!
            # read in our soft labels and use to replace any of the one hot 
            our_soft_labels_file="./data/raw_elicitation_data_ours.json"
            with open(our_soft_labels_file, "r") as f: 
                self.elicitation_data = json.load(f) 
                    
            # extract stats over all of our soft labels  
            # based on whether the equiv cifar10h label is high or low ent
            high_ent_stats = {"top_prob": [], "n_also_poss": []}
            low_ent_stats = {"top_prob": [], "n_also_poss": []}
            self.semantic_sims = get_semantic_sim_matrix(self.classes)
            self.ent_thresh=0.25 # ent thresh used was 0.25 in construction
            # here, use all of the labels we have available to get these stats for the generative model
            for example_idx, elicited_info in self.elicitation_data.items():
                example_idx = int(example_idx)

                cifar10h_ent = entropy(self.human_label_counts[example_idx]/np.sum(self.human_label_counts[example_idx]))
                n_tot_annotators= len(elicited_info)
                elicited_annotator_info = np.array(elicited_info) # take all 
                # just use top2 clamp version for now
                annotator_labels = []
                for single_annotator_info in elicited_annotator_info:
                    annotator_label = construct_elicited_soft_label(single_annotator_info, self.class2idx, self.idx2class, include_top_2 = True, redist="clamp",
                            redist_factor=0.1, semantic_sims=False) 
                    annotator_label = np.array(annotator_label)
                    
                    top_prob = annotator_label.max()
                    n_also_poss = np.sum(annotator_label != 0) - 1 # to account for the already spec most prob
                    
                    if cifar10h_ent >= self.ent_thresh: 
                        ent_type="high"
                        high_ent_stats["top_prob"].append(top_prob)
                        high_ent_stats["n_also_poss"].append(n_also_poss) 
                    else: 
                        ent_type="low"
                        low_ent_stats["top_prob"].append(top_prob)
                        low_ent_stats["n_also_poss"].append(n_also_poss)  
            # fit beta dist for top prob (want [0,1] b/c prob)
            self.low_ent_a, self.low_ent_b = fit_beta_params(low_ent_stats["top_prob"],ent_type)
            self.high_ent_a, self.high_ent_b = fit_beta_params(high_ent_stats["top_prob"],ent_type)
            # use the distribution of poss -- sample freq, since discrete
            # note: valid values b/w [0, 9] (can't have all 10 imposs, b/c must select a class)
            self.low_poss = low_ent_stats["n_also_poss"]
            self.high_poss = high_ent_stats["n_also_poss"]
            self.low_ent_n_poss_probs = get_n_poss_dist(low_ent_stats["n_also_poss"])
            self.high_ent_n_poss_probs = get_n_poss_dist(high_ent_stats["n_also_poss"])
                        
            num_annotators_sample=6 # for now
            self.labels = [self.create_simulated_our_label(counts, num_annotators_sample) for counts in self.human_label_counts]

            # BEFORE: 
            # self.labels = self.human_soft_labels


        else:
            self.labels = self.img_dataset.targets#[self.img_dataset[i][1] for i in range(data_size)]

        self.transform = transform
        self.label_transform = label_transform

        self.label_method = label_method
        # load in hill data
        # {filename: participant data, ...}
        # help from: https://stackoverflow.com/questions/16573332/jsondecodeerror-expecting-value-line-1-column-1-char-0
        with open(hill_mix_pth, "r") as f: 
            self.hill_data = json.loads(f.read())

        with open(hill_mix_soft_label_pth, "r") as f: 
            self.hill_mix_soft_labels = json.loads(f.read())

        # before doing img mixing, need to apply transform to all images
        # apply transform to image like in: https://github.com/jcpeterson/cifar10-human-experiments/blob/efadada2c1adc6bcdc8d86aca7e542564ff2980e/pytorch_image_classification/dataloader_c10h.py#L42
        if self.transform is not None: 
            self.imgs = [self.transform(img) for img in self.imgs]

        if split == "train" and not no_aug_set: 
            # include hill relabelings
            # load and construct appropriate label
            self.hill_mix_filenames = []
            self.hill_mix_imgs = []
            self.hill_mix_labels = []
            for filename, human_labelings in self.hill_data.items(): 

                if rem_hill_mix_soft and filename in self.hill_mix_soft_labels: continue 

                label_vec = np.zeros([self.num_classes]) # begin a new label vector 
                _, mixing_coeff, label1, id1_str, label2, id2_str = filename.split("_")
                label_idx1 = self.class2idx[label1]
                label_idx2 = self.class2idx[label2] # use to index into the K-dim vec

                # create the same image that participants were shown -- just mixup by coeff
                mixing_coeff = float(mixing_coeff)
                id1 = int(id1_str) # indices for the examples
                id2 = int(id2_str.split(".png")[0]) # remove trailing file tag 
                mixed_x = mixing_coeff * self.imgs[id1] + (1-mixing_coeff) * self.imgs[id2]

                # based on labeling method, modify 
                if label_method == "avg": 
                    # take avg mixing coeff and apply
                    lam = np.mean(human_labelings["Predicted Mixing Factor"])
                    label_vec[label_idx1] = lam
                    label_vec[label_idx2] = (1-lam) 

                    self.hill_mix_imgs.append(mixed_x)
                    self.hill_mix_labels.append(label_vec)
                    self.hill_mix_filenames.append(filename)

                 # include a random label that sums to one 
                if label_method == "random": 
                    # help from: https://stackoverflow.com/questions/18659858/generating-a-list-of-random-numbers-summing-to-1
                    label_vec = np.array([random.random() for i in range(self.num_classes)])
                    label_vec = label_vec/np.sum(label_vec)

                    self.hill_mix_imgs.append(mixed_x)
                    self.hill_mix_labels.append(label_vec)
                    self.hill_mix_filenames.append(filename)


                elif label_method == "perAnnotatorConfSmooth": 
                    # smooth based on confidence
                    # for now, do this on a per-annotator basis
                    # if conf = 0 --> uniform, conf = 1 --> use their relabeling exactly 
                    annotator_label_vecs = []
                    pred_mix_coeffs= human_labelings["Predicted Mixing Factor"]
                    pred_confs = human_labelings["Predicted Confidence"]
                    for lam, pred_conf in zip(pred_mix_coeffs,pred_confs): 
                        label_vec[label_idx1] = lam
                        label_vec[label_idx2] = (1-lam)

                        # todo: play with the best label smoothing!! 
                        # help from nlp work and additive smoothing: https://vitalflux.com/quick-introduction-smoothing-techniques-language-models/
                        v = label_vec * self.num_classes # to have more in "count" format
                        # a = 100
                        # b = 0.0005#0.000001
                        a = smoother_a_param
                        b = smoother_b_param
                        conf_smoother = lambda x : a * (b ** x)

                        s = np.sum(v) 
                        conf_scale = conf_smoother(pred_conf)
                        smoothed_vec = (v + conf_scale) / (s + self.num_classes * conf_scale)

                        annotator_label_vecs.append(smoothed_vec)

                    self.hill_mix_imgs.extend([mixed_x for _ in range(len(annotator_label_vecs))]) # per annotator
                    self.hill_mix_labels.extend(annotator_label_vecs)
                    self.hill_mix_filenames.append([filename for _ in range(len(annotator_label_vecs))])

                                
                                
                elif label_method == "confSmooth": 
                    # smooth based on confidence
                    # aggregate over annotators
                    # if conf = 0 --> ~uniform, conf = 1 --> use their relabeling exactly 
                    annotator_label_vecs = []
                    pred_mix_coeffs= human_labelings["Predicted Mixing Factor"]
                    pred_confs = human_labelings["Predicted Confidence"]

                    # take avg mixing coeff and apply
                    lam = np.mean(human_labelings["Predicted Mixing Factor"])
                    pred_conf = np.mean(human_labelings["Predicted Confidence"])
                    label_vec[label_idx1] = lam
                    label_vec[label_idx2] = (1-lam) 

                    # todo: play with the best label smoothing!! 
                    # help from nlp work and additive smoothing: https://vitalflux.com/quick-introduction-smoothing-techniques-language-models/
                    v = label_vec * self.num_classes # to have more in "count" format
                    # a = 100
                    # b = 0.0005#0.000001
                    a = smoother_a_param
                    b = smoother_b_param
                    conf_smoother = lambda x : a * (b ** x)

                    s = np.sum(v) 
                    conf_scale = conf_smoother(pred_conf)
                    smoothed_vec = (v + conf_scale) / (s + self.num_classes * conf_scale)

                    annotator_label_vecs.append(smoothed_vec)

                    self.hill_mix_imgs.append(mixed_x)
                    self.hill_mix_labels.append(smoothed_vec)
                    self.hill_mix_filenames.append(filename)

                elif label_method == "uniform":
                    # baseline
                    # just label each example with the uniform class distribution
                    # note: no actual HILL here
                    self.hill_mix_imgs.append(mixed_x) 
                    self.hill_mix_labels.append(np.ones([self.num_classes])/self.num_classes)
                elif label_method == "perAnnotatorNoConf":
                    # similar to confidence smoothing in that each annotator's label is provided as a separate example
                    # e.g., redundancy for x (but possibly different y's)
                    # but now, no use of the predicted confidence -- just use pred mix coeff
                    annotator_label_vecs = []
                    pred_mix_coeffs= human_labelings["Predicted Mixing Factor"]
                    for lam in pred_mix_coeffs: 
                        label_vec[label_idx1] = lam
                        label_vec[label_idx2] = (1-lam)
                        annotator_label_vecs.append(label_vec)

                    self.hill_mix_imgs.extend([mixed_x for _ in range(len(annotator_label_vecs))]) # per annotator
                    self.hill_mix_labels.extend(annotator_label_vecs)
                    self.hill_mix_filenames.append([filename for _ in range(len(annotator_label_vecs))])
                elif label_method == "generatingMixFactor":
                    # always just use the same underlying 
                    label_vec[label_idx1] = mixing_coeff
                    label_vec[label_idx2] = (1-mixing_coeff) 

                    self.hill_mix_imgs.append(mixed_x)
                    self.hill_mix_labels.append(label_vec)
                    self.hill_mix_filenames.append(filename)
                elif label_method == "genMixOverSameSoft": 
                    # only include if in the same set of filenames that we have soft labels for
                    if filename in self.hill_mix_soft_labels:
                        label_vec[label_idx1] = mixing_coeff
                        label_vec[label_idx2] = (1-mixing_coeff) 
                        self.hill_mix_imgs.append(mixed_x)
                        self.hill_mix_labels.append(label_vec)
                        self.hill_mix_filenames.append(filename)
                elif label_method == "generatingMixFactorRemoveUnc": 
                    # use mixup labeling, but remove if low confidence
                    pred_confs = human_labelings["Predicted Confidence"]
                    avg_conf = np.mean(pred_confs)
                    conf_thresh = 0.25 
                    if avg_conf > conf_thresh: 
                        # only include this example if sufficiently high avg confidence
                        label_vec[label_idx1] = mixing_coeff
                        label_vec[label_idx2] = (1-mixing_coeff) 
                        self.hill_mix_imgs.append(mixed_x)
                        self.hill_mix_labels.append(label_vec)
                        self.hill_mix_filenames.append(filename)
                elif label_method == "generatingMixFactorRemoveRelabel": 
                    # use mixup labeling, but remove if low confidence
                    avg_relabel = np.abs(human_labelings["Re-Label Diff"])
                    thresh = 0.25 
                    if avg_relabel < thresh: 
                        # only include this example if not largely relabeled
                        label_vec[label_idx1] = mixing_coeff
                        label_vec[label_idx2] = (1-mixing_coeff) 
                        self.hill_mix_imgs.append(mixed_x)
                        self.hill_mix_labels.append(label_vec)
                        self.hill_mix_filenames.append(filename)
                elif label_method == "useOurSoft": 
                    if filename in self.hill_mix_soft_labels: 
                        label_vec = np.array(self.hill_mix_soft_labels[filename])
                    # else: 
                    #     # use typical mixup label
                    #     label_vec[label_idx1] = mixing_coeff
                    #     label_vec[label_idx2] = (1-mixing_coeff) 

                        self.hill_mix_imgs.append(mixed_x)
                        self.hill_mix_labels.append(label_vec)
                        self.hill_mix_filenames.append(filename)
        
        if just_hill_mix_soft: 
            # overwrite and just have the mixed images with our elicited filenames
            soft_x = []
            soft_files = []
            soft_labels = []
            for filename, agg_human_soft_label in self.hill_mix_soft_labels.items(): 
                _, mixing_coeff, label1, id1_str, label2, id2_str = filename.split("_")

                # create the same image that participants were shown -- just mixup by coeff
                mixing_coeff = float(mixing_coeff)
                id1 = int(id1_str) # indices for the examples
                id2 = int(id2_str.split(".png")[0]) # remove trailing file tag 
                mixed_x = mixing_coeff * self.imgs[id1] + (1-mixing_coeff) * self.imgs[id2]
                soft_files.append(filename)
                soft_x.append(mixed_x)
                soft_labels.append(agg_human_soft_label)
            self.filenames = soft_files
            self.imgs = soft_x
            self.labels = soft_labels
        else: 
            # typical data used! 
            # optionally use specified train/test split
            # modified from Uma for implementing:
            # note: do this after getting the mixup examples, since indices are matched off the full set
            if not use_all_data:
                if split == "train": 
                    indices = np.load(f"{split_idx_pth}/train_indices.npy").tolist()
                    # if we are trying to hold out the mixup examples we have soft labels for, remove those endpoint ids
                    if rem_hill_mix_soft: 
                        rem_indices = set()
                        for filename in self.hill_mix_soft_labels.keys(): 
                            mixing_coeff, label1, id1_str, label2, id2_str = filename.split("_")[1:]
                            id1 = int(id1_str)
                            id2 = int(id2_str.split(".png")[0])
                            rem_indices.add(id1)
                            rem_indices.add(id2)
                        indices = [idx for idx in indices if idx not in rem_indices]
                else: 
                    # use test indices
                    indices = np.load(f"{split_idx_pth}/test_indices.npy").tolist()

                # non-numpy filtering help from: 
                # self.imgs = [i for (i, v) in zip(list_a, filter) if v]
                self.imgs = list(np.array(self.imgs)[indices])
                self.labels = list(np.array(self.labels)[indices])

            self.imgs = self.imgs
            self.labels=self.labels

        if split == "train" and not no_aug_set: 

            # if aug size is -1, use all the examples we have labelings for (hill_mix_imgs)
            # if aug size < |self.hill_mix_imgs| --- downsample
            # o.w., create more samples, for now, using the generating mix factor approach
            n_aug_imgs = len(self.hill_mix_imgs)
            if aug_size != -1: 
                if aug_size < n_aug_imgs: # downsample
                    self.downsampled_idxs = np.random.choice(n_aug_imgs, aug_size, replace=False)
                    self.hill_mix_imgs = np.array(self.hill_mix_imgs)[self.downsampled_idxs]
                    self.hill_mix_labels = np.array(self.hill_mix_labels)[self.downsampled_idxs]
                    self.hill_mix_filenames = np.array(self.hill_mix_filenames)[self.downsampled_idxs]
                elif aug_size > n_aug_imgs: # upsample
                    # create more samples -- for now, using the generating mix factor approach, but can explore others
                    # use our same set-up for now as well (e.g., selecting from a fixed set of mixing coeffs)
                    coeffs = [0.1, 0.25, 0.5, 0.75, 0.9]
                    for sample_idx in range(aug_size - n_aug_imgs): 
                        mixing_coeff = np.random.choice(coeffs)
                        # check to make sure we don't already have this, though unlikely
                        # also ensure samples are from different classes
                        sample_okay = False
                        while not sample_okay: 
                            [id1, id2] = np.random.choice(range(len(self.imgs)), 2, replace=False) 
                            label_1 = self.labels[id1]
                            label_2 = self.labels[id2] 
                            if label1 == label2: continue 
                            # create what the file name would be according to other approach
                            label_txt_1 = self.idx2class[label_1]
                            label_txt_2 = self.idx2class[label_2]
                            # map our index into the original indexes in the dataset (using indices obj)
                            # since the original mixup imgs were saved using the full indices 
                            adjusted_idx_1 = indices[id1]
                            adjusted_idx_2 = indices[id2]
                            synth_filename = f"mixed_{mixing_coeff}_{label_txt_1}_{adjusted_idx_1}_{label_txt_2}_{adjusted_idx_2}.png"
                            if synth_filename in self.hill_mix_filenames: continue 

                            # otherwise, selected pair is okay -- create and add to set
                            mixed_x = mixing_coeff * self.imgs[id1] + (1-mixing_coeff) * self.imgs[id2]
                            label_vec = np.zeros([self.num_classes]) # begin a new label vector 
                            label_vec[label_1] = mixing_coeff
                            label_vec[label_2] = 1-mixing_coeff
                            self.hill_mix_imgs.append(mixed_x)
                            self.hill_mix_labels.append(label_vec)
                            self.hill_mix_filenames.append(f"aug{synth_filename}")
                            sample_okay = True 

            self.filenames = [f"original_{self.idx2class[class_idx]}_{idx}" for idx, class_idx in zip(range(len(self.imgs)), self.img_dataset.targets)]
            self.filenames.extend(self.hill_mix_filenames)

            self.imgs.extend(self.hill_mix_imgs)
            self.labels.extend(self.hill_mix_labels)
            
        # self.imgs = self.hill_mix_imgs
        # self.labels = self.hill_mix_labels

        # print("new size: ", len(self.imgs), len(self.hill_mix_imgs))

    def __len__(self):
        '''
        Return num data points
        '''
        return len(self.imgs) # num data points

    def get_class_per_idx(self): 
        '''
        Get semantically meaningful class labels associated with each example index
        Returns a map of {CIFAR-10/H example idx: associated label (txt), ...}
        '''
        idx2class = {v: k.capitalize() for k,v in self.img_dataset.class_to_idx.items()}
        return idx2class

    def get_classes(self): 
        '''
        Return semantically meaningful class labels 
        As an (ordered) list
        '''
        return [class_name.capitalize() for class_name in self.img_dataset.classes]

    def create_simulated_our_label(self, cifar10h_counts, M_subsample): 

        cifar10h_ent = entropy(cifar10h_counts/np.sum(cifar10h_counts))
        if cifar10h_ent >= self.ent_thresh: ent_type = "high"
        else: ent_type="low"

        annotator_labels = [] 
        for label_id, num_ann_pred in enumerate(cifar10h_counts):
            for _ in range(num_ann_pred): 
                # convert to one of ours
                annotator_labels.append(self.sample_our_generative_model(label_id, ent_type))
        
        # keep de-aggregated for now? 
        if M_subsample <= len(annotator_labels): 
            sampled_annotator_idxs = np.random.choice(list(range(len(annotator_labels))), M_subsample, replace=False)
            sampled_annotators = np.array(annotator_labels)[sampled_annotator_idxs]
        else: sampled_annotators = np.array(annotator_labels)

        return np.mean(sampled_annotators,axis=0)
        

    def sample_our_generative_model(self, most_prob_class, ent_type="low"): 
        sampled_label = np.zeros(self.num_classes)
        if ent_type == "low": 
            sampled_top_prob = np.random.beta(self.low_ent_a, self.low_ent_b)
            # help from: https://stackoverflow.com/questions/11373192/generating-discrete-random-variables-with-specified-weights-using-scipy-or-numpy
            num_other_poss = np.random.choice(list(range(len(self.low_ent_n_poss_probs))), 1, p=self.low_ent_n_poss_probs)[0]
        else:
            sampled_top_prob = np.random.beta(self.high_ent_a, self.high_ent_b)
            num_other_poss = np.random.choice(list(range(len(self.high_ent_n_poss_probs))), 1, p=self.high_ent_n_poss_probs)[0]
        sampled_label[most_prob_class] = sampled_top_prob
        # sort the alternative labels based on semantic distance
        sorted_semantic_sims = []
        for idx in range(self.num_classes): 
            if idx != most_prob_class: sorted_semantic_sims.append((self.semantic_sims[most_prob_class, idx], idx))
        sorted_semantic_sims = sorted(sorted_semantic_sims, key=lambda x:x[0], reverse=True)
        # keep only based on num other poss
        alt_classes = [x[1] for x in sorted_semantic_sims[:num_other_poss]]
        # second most prob label based on how much is leftover and num other poss
        if num_other_poss != 0: 
            mass_leftover = 1-sampled_top_prob
            sampled_second_prob = np.random.uniform((1/num_other_poss)*mass_leftover, mass_leftover)
            sampled_label[alt_classes[0]] = sampled_second_prob

            # spread uniformly over the remaining 
            alt_classes = alt_classes[1:]
            mass_leftover = 1-np.sum(sampled_label)
            if len(alt_classes) != 0 and mass_leftover < 1.0: 
                for alt_class in alt_classes: 
                    sampled_label[alt_class] = 1/len(alt_classes) * mass_leftover


        return sampled_label / np.sum(sampled_label) # ensures sums to 1 just in case

    def __getitem__(self, idx):
        '''
        Combine the imgs from cifar10 loader with human generated soft labels
        Assumes these images have transform applied (per cifar10-dataloader init)
        Returns img with target
        ''' 
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        # print("index: ", idx, " imgs: ", len(self.imgs), " labels: ", len(self.labels))

        img = self.imgs[idx]
        label = torch.tensor(self.labels[idx])

        # # apply transform to image like in: https://github.com/jcpeterson/cifar10-human-experiments/blob/efadada2c1adc6bcdc8d86aca7e542564ff2980e/pytorch_image_classification/dataloader_c10h.py#L42
        # if self.transform: 
        #     img = self.transform(img)

        # print("label: ", label)

        # ensure all labels are in K class vector form if not already
        if len(label.shape) == 0: # index tensor 
            label = F.one_hot(label, num_classes=self.num_classes) # https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html


        # handle type conversions
        # help from: https://stackoverflow.com/questions/56741087/how-to-fix-runtimeerror-expected-object-of-scalar-type-float-but-got-scalar-typ
        # print("updated label: ", label.detach().numpy().astype(np.float32),label.dtype)
        label = torch.FloatTensor(label.detach().numpy().astype(np.float32))
        # print("rev label: ", label, label.dtype)

        # img = self.idx2img[idx][0] # already a tensor from init [1, 3, 32, 32]
        # # need type long: https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-long-but-got-scalar-type-float-when-using-crossentropyloss/30542/2
        # label = torch.tensor(self.human_soft_labels[idx])#,  dtype=torch.long)

        return (img, label)


class CIFAR10HU(Dataset):
    """
    Data loader for Peterson CIFAR-10H, with optional single-annotator uncertainty (simulation)
    CIFAR-10H data from Peterson et al: https://github.com/jcpeterson/cifar-10h
    modified dataloader from example: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, human_soft_labels_file="./data/cifar10h-counts.npy", img_dir='~/data',
                    transform=None, label_transform=None, split_idx_pth="./data/", split="train",
                    per_annotator_uncertainty_level=0,
                    redistribution_method = "uniform",
                    conf_mat_file="/home/kmc61/cifar10-human-experiments/confusion_matrices/matrices/Human_all_cms.p",
                    label_method="cifar10h", our_soft_labels_file="./data/raw_elicitation_data_ours.json",
                    use_all_cifar10h=False, num_annotators_sample=-1, annotator_subsample_seed=0,
                    redist_level=0.1, data_split_seed=7, num_examples_holdout=100, subsample_soft=-1, 
                    use_per_annotator=False, use_cifar10h_base=False,use_ls_base=False, ls_smooth_amt=0.01, use_our_base=False
                    ):
        """
        Args:
            human_soft_labels_file (string, optional): Path to file with human soft labels (cifar10h)
            img_dir (string, optional): Pth to directory with CIFAR-10 (test) images used by Peterson, et al.
            transform (callable, optional): Optional transform to be applied
                on image.
            label_transform: Optional transform over label space, e.g., smoothing.
            per_annotator_uncertainty_level: Amount of noise to add to the labels before aggregation to each annotator
            redistribution_method: How to spread the extra mass 
            modified from example: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        """
         # shape = [num_data_points, num_classes], e.g. [10,000, 10]
        self.human_label_counts = np.load(human_soft_labels_file)

        use_conf_mat = False 

        self.use_per_annotator=use_per_annotator

        if use_conf_mat: # hack
            with open(conf_mat_file, "rb") as f: 
                # see: https://github.com/jcpeterson/cifar10-human-experiments/blob/master/confusion_matrices/Human%20allconfusion_matrix.png
                cm_data = pickle.load(f)["softmax"]


            # normalize (dist given human choice)
            # matrix norm help from: https://stackoverflow.com/questions/8904694/how-to-normalize-a-2-dimensional-numpy-array-in-python-less-verbose
            # norm_factor = cm_data.sum(axis=0)
            # cm_data = cm_data / norm_factor[np.newaxis, :]
            # self.human_confusion_matrix = cm_data
            norm_factor = cm_data.sum(axis=1)
            cm_data = cm_data / norm_factor[:, np.newaxis]
            self.human_confusion_matrix = cm_data

        # read in our soft labels and use to replace any of the one hot 
        with open(our_soft_labels_file, "r") as f: 
            self.elicitation_data = json.load(f) 

        # extract from our elicitation data the indexes of the images queried
        self.relabed_idxs = set([int(example_idx) for example_idx in self.elicitation_data.keys()])

        self.img_dir = img_dir
        if img_dir is None: 
            # re-download cifar10 test set
            download_orig_cifar10 = True
            img_dir = "~/data" # save to
        else: download_orig_cifar10 = False
        self.img_dataset = datasets.CIFAR10(root=img_dir, train=False, download=download_orig_cifar10,
                        transform=transform) 

        # self.imgs = self.img_dataset.data
        self.semantic_sims=None

        data_size = len(self.img_dataset)
        # modified from Uma et al: https://github.com/AlexandraUma/dali-learning-with-disagreement/blob/main/iccifar10/soft_loss.py 
        self.imgs = [self.img_dataset[i][0] for i in range(data_size)]
        self.orig_hard_labels = [self.img_dataset[i][1] for i in range(data_size)]

        # extract names of classes
        self.classes = self.get_classes()
        # map from names of classes to indices
        self.class2idx = {class_name: idx for idx,class_name in enumerate(self.classes)}
        self.idx2class = {idx: class_name for idx,class_name in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        if subsample_soft != -1: 
            # downsample the poss examples that we replace the hard labels w/ soft labels for 
            np.random.seed(data_split_seed)
            random.seed(data_split_seed)
            relabel_examples = set(random.sample(self.relabed_idxs, subsample_soft)) # w/o replacement
        else: relabel_examples = set(self.relabed_idxs)
        np.random.seed(annotator_subsample_seed)
        random.seed(annotator_subsample_seed)

        if use_cifar10h_base: 
            # todo: change this to have M-dependent
            self.labels = [counts/np.sum(counts) for counts in self.human_label_counts]
        elif use_ls_base:
            ls_alpha = 0.05 # from https://proceedings.neurips.cc/paper/2019/file/f1748d6b0fd9d439f71450117eba2725-Paper.pdf and https://proceedings.neurips.cc/paper/2019/file/f1748d6b0fd9d439f71450117eba2725-Paper.pdf 
            # corroborated w/ cross-val            
            self.labels  = [create_smoothed_label(hard_label_class, num_classes=self.num_classes, smoothing_factor=ls_alpha) for hard_label_class in self.orig_hard_labels]
        elif use_our_base: 


            # extract stats over all of our soft labels  
            # based on whether the equiv cifar10h label is high or low ent
            high_ent_stats = {"top_prob": [], "n_also_poss": []}
            low_ent_stats = {"top_prob": [], "n_also_poss": []}
            self.semantic_sims = get_semantic_sim_matrix(self.classes)
            self.ent_thresh=0.25 # ent thresh used was 0.25 in construction
            # here, use all of the labels we have available to get these stats for the generative model
            for example_idx, elicited_info in self.elicitation_data.items():
                example_idx = int(example_idx)

                cifar10h_ent = entropy(self.human_label_counts[example_idx]/np.sum(self.human_label_counts[example_idx]))
                n_tot_annotators= len(elicited_info)
                elicited_annotator_info = np.array(elicited_info) # take all 
                # just use top2 clamp version for now
                annotator_labels = []
                for single_annotator_info in elicited_annotator_info:
                    annotator_label = construct_elicited_soft_label(single_annotator_info, self.class2idx, self.idx2class, include_top_2 = True, redist="clamp",
                            redist_factor=redist_level, semantic_sims=False) 
                    annotator_label = np.array(annotator_label)
                    
                    top_prob = annotator_label.max()
                    n_also_poss = np.sum(annotator_label != 0) - 1 # to account for the already spec most prob
                    
                    if cifar10h_ent >= self.ent_thresh: 
                        ent_type="high"
                        high_ent_stats["top_prob"].append(top_prob)
                        high_ent_stats["n_also_poss"].append(n_also_poss) 
                    else: 
                        ent_type="low"
                        low_ent_stats["top_prob"].append(top_prob)
                        low_ent_stats["n_also_poss"].append(n_also_poss)  
            # fit beta dist for top prob (want [0,1] b/c prob)
            self.low_ent_a, self.low_ent_b = fit_beta_params(low_ent_stats["top_prob"],ent_type)
            self.high_ent_a, self.high_ent_b = fit_beta_params(high_ent_stats["top_prob"],ent_type)
            # use the distribution of poss -- sample freq, since discrete
            # note: valid values b/w [0, 9] (can't have all 10 imposs, b/c must select a class)
            self.low_poss = low_ent_stats["n_also_poss"]
            self.high_poss = high_ent_stats["n_also_poss"]
            self.low_ent_n_poss_probs = get_n_poss_dist(low_ent_stats["n_also_poss"])
            self.high_ent_n_poss_probs = get_n_poss_dist(high_ent_stats["n_also_poss"])







            # sample from our generative model
            # generated using cifar10h labels, and a function of num downsample 
            self.labels = [self.create_simulated_our_label(counts, num_annotators_sample) for counts in self.human_label_counts]
        else: self.labels = self.orig_hard_labels

        # if use_human_soft: 
        #     self.counts = self.human_label_counts
        # else: self.labels = self.img_dataset.targets

        self.transform = transform

        # before doing img mixing, need to apply transform to all images
        # apply transform to image like in: https://github.com/jcpeterson/cifar10-human-experiments/blob/efadada2c1adc6bcdc8d86aca7e542564ff2980e/pytorch_image_classification/dataloader_c10h.py#L42
        # if self.transform is not None: 
        #     self.imgs = [self.transform(img) for img in self.imgs]

        # self.labels = self.img_dataset.targets # hard targets

        # optionally replace with other labeling methods
        # e.g., ours, peterson's, (or peterson's over the full set)


        # keep track of individ annotator labels if using on our the humna-based soft label methods
        self.individ_annotator_labels = {}

        self.label_method = label_method

        if "ours" in label_method: 

            # extract properties of transform from label method
            if "Top2" in label_method: include_top_2 = True
            else: include_top_2 = False
            if "SemanticClamp" in label_method: redist="semanticClamp"
            elif "Semantic" in label_method: redist="semantic"
            elif "Clamp" in label_method: redist = "clamp"
            elif "Uniform" in label_method: redist="uniform"
            elif "None" in label_method: redist="none"
            elif "NLS" in label_method: redist="nls"
            else: redist="none"

            if "semantic" in redist:
                if self.semantic_sims is not None: semantic_sims = self.semantic_sims 
                else: semantic_sims = get_semantic_sim_matrix(self.classes)
            else: semantic_sims = None # don't consider

            for example_idx, elicited_info in self.elicitation_data.items():
                example_idx = int(example_idx)

                if example_idx not in relabel_examples: continue 

                n_tot_annotators= len(elicited_info)
                # possibly downsample number of annotators
                # if num_annotators_sample != -1 or num_annotators_sample < n_tot_annotators:
                if num_annotators_sample != -1 and num_annotators_sample < n_tot_annotators:
                    elicited_annotator_info = np.array(elicited_info)[np.random.choice(list(range(n_tot_annotators)), num_annotators_sample, replace=False)]
                else: elicited_annotator_info = np.array(elicited_info) # take all 
                # transform into labels, and take average to aggregate (for now) if M > 1 
                annotator_labels = []
                for single_annotator_info in elicited_annotator_info:
                    if not "SimSelect2" in label_method: 
                        annotator_label = construct_elicited_soft_label(single_annotator_info, self.class2idx, self.idx2class, include_top_2 = include_top_2, redist=redist,
                            redist_factor=redist_level, semantic_sims=semantic_sims)
                    else: 
                        # run the top 2 simulated cifar-10h select
                        annotator_label = create_cifar10h_sim2(single_annotator_info, self.class2idx)


                    if "Smooth" in label_method: 
                        annotator_label = annotator_label * (1-ls_smooth_amt) + np.ones([self.num_classes]) * ls_smooth_amt
                    annotator_labels.append(annotator_label)
                annotator_labels = np.array(annotator_labels)
                label = np.mean(annotator_labels,axis=0)
                self.labels[example_idx] = label
                self.individ_annotator_labels[example_idx] = annotator_labels

            # for example_idx, elicited_info in self.elicitation_data.items(): 
            #     self.labels[int(example_idx)] = elicited_info[label_method]
        elif "cifar10h" in label_method: 
            # convert counts into labels -- optionally subsample annotators
            for example_idx, counts in enumerate(self.human_label_counts):
                if example_idx in relabel_examples or use_all_cifar10h: # if use all, just replace for every label
                    n_tot_annotators= np.sum(counts) 
                    # extract the "original" annotators' labels (we know that they're just the hard labels)
                    # then sample from this
                    # note: hacky -- find other way to "reverse engineer" peterson labelers (alternative is weighted sampling, but this is "safer" to start)
                    annotator_labels = [] 
                    for label_id, num_ann_pred in enumerate(counts):
                        for _ in range(num_ann_pred): 
                            annotator_label = np.zeros([self.num_classes])
                            annotator_label[label_id] = 1.0

                            # if "Smooth" in label_method: 
                            #     # smooth this label
                            #     annotator_label = annotator_label * (1-redist_level) + np.ones([self.num_classes]) * redist_level

                            annotator_labels.append(annotator_label)
                    # check if we're subsampling (if -1, use all annotations)
                    if num_annotators_sample != -1:
                        # subsample and optionally aggregate if num_annotators_sample > 1 
                        sampled_annotator_idxs = np.random.choice(list(range(len(annotator_labels))), num_annotators_sample, replace=False)
                        sampled_annotators = np.array(annotator_labels)[sampled_annotator_idxs]
                        label = np.mean(sampled_annotators, axis=0)
                    else: 
                        # use all annotations
                        # label = counts / n_tot_annotators 
                        sampled_annotators = np.array(annotator_labels) # just all annotators
                        label = np.mean(sampled_annotators, axis=0)

                    if "Smooth" in label_method: 
                        label = label * (1-ls_smooth_amt) + np.ones([self.num_classes]) * ls_smooth_amt
                    self.labels[example_idx] = label
                    self.individ_annotator_labels[example_idx] = sampled_annotators


        elif label_method == "uniform": 
            for example_idx, elicited_info in self.elicitation_data.items():
                example_idx = int(example_idx)
                self.labels[example_idx] = np.ones([self.num_classes]) * 1/self.num_classes 

        elif label_method == "random": 
            for example_idx, elicited_info in self.elicitation_data.items():
                example_idx = int(example_idx)
                # help from: https://stackoverflow.com/questions/18659858/generating-a-list-of-random-numbers-summing-to-1
                rand_vec = np.array([random.random() for i in range(self.num_classes)])
                rand_vec = rand_vec/np.sum(rand_vec)
                self.labels[example_idx] = rand_vec

        elif label_method == "labelSmooth": 
            ls_alpha = 0.05 # from Table 1: https://proceedings.neurips.cc/paper/2019/file/f1748d6b0fd9d439f71450117eba2725-Paper.pdf and https://proceedings.neurips.cc/paper/2019/file/f1748d6b0fd9d439f71450117eba2725-Paper.pdf 
            
            # ls_alpha = ls_smooth_amt
            
            for example_idx, elicited_info in self.elicitation_data.items():
                example_idx = int(example_idx)
                hard_label_class = self.orig_hard_labels[example_idx]
                smoothed_label = create_smoothed_label(hard_label_class, num_classes=self.num_classes, smoothing_factor=ls_alpha)
                # hard_label = np.zeros([self.num_classes])
                # hard_label[label_class] = 1.0 
                # # smoothed_label = np.ones([self.num_classes]) * (ls_alpha/(self.num_classes - 1)) # apply smoother to all classes which are not the hard label class
                # # smoothed_label[hard_label] = 1.0 * (1-ls_alpha)
                # smoothed_label = hard_label * (1-ls_alpha) + np.ones([self.num_classes]) * ls_alpha
                self.labels[example_idx] = smoothed_label

        elif label_method == "baseline": 
            for example_idx, elicited_info in self.elicitation_data.items():
                example_idx = int(example_idx)
                self.labels[example_idx] = self.img_dataset[example_idx][1]


        # optionally use specified train/test split
        # modified from Uma for implementing:
        # note: do this after getting the mixup examples, since indices are matched off the full set
        if split_idx_pth is not None and split is not None:
            if split == "train": 
                indices = np.load(f"{split_idx_pth}/train_indices.npy").tolist()
            elif split == "trainSub": 
                np.random.seed(data_split_seed)
                random.seed(data_split_seed)
                indices = np.load(f"{split_idx_pth}/train_indices.npy").tolist()
                holdout = random.sample(list(self.relabed_idxs), num_examples_holdout) # w/o replacement
                indices = [idx for idx in indices if idx not in holdout]
            elif split == "testSub": 
                np.random.seed(data_split_seed)
                random.seed(data_split_seed)
                indices = np.load(f"{split_idx_pth}/train_indices.npy").tolist()
                holdout = random.sample(list(self.relabed_idxs), num_examples_holdout) # w/o replacement
                indices = list(holdout)
            else: 
                # use test indices
                indices = np.load(f"{split_idx_pth}/test_indices.npy").tolist()

            # non-numpy filtering help from: 
            # self.imgs = [i for (i, v) in zip(list_a, filter) if v]
            self.imgs = list(np.array(self.imgs)[indices])
            self.labels = list(np.array(self.labels)[indices])
            self.indices = indices 
        else: self.indices = list(range(0, len(self.imgs)))

        # map from the sampled indices back to the "original" indices
        self.index_converter = {new_idx: orig_idx for new_idx, orig_idx in enumerate(indices)}

    def __len__(self):
        '''
        Return num data points
        '''
        return len(self.imgs) # num data points

    def get_class_per_idx(self): 
        '''
        Get semantically meaningful class labels associated with each example index
        Returns a map of {CIFAR-10/H example idx: associated label (txt), ...}
        '''
        idx2class = {v: k.capitalize() for k,v in self.img_dataset.class_to_idx.items()}
        return idx2class

    def get_classes(self): 
        '''
        Return semantically meaningful class labels 
        As an (ordered) list
        '''
        return [class_name.capitalize() for class_name in self.img_dataset.classes]

    def create_simulated_our_label(self, cifar10h_counts, M_subsample): 

        cifar10h_ent = entropy(cifar10h_counts/np.sum(cifar10h_counts))
        if cifar10h_ent >= self.ent_thresh: ent_type = "high"
        else: ent_type="low"

        annotator_labels = [] 
        for label_id, num_ann_pred in enumerate(cifar10h_counts):
            for _ in range(num_ann_pred): 
                # convert to one of ours
                annotator_labels.append(self.sample_our_generative_model(label_id, ent_type))
        
        # keep de-aggregated for now? 
        if M_subsample <= len(annotator_labels): 
            sampled_annotator_idxs = np.random.choice(list(range(len(annotator_labels))), M_subsample, replace=False)
            sampled_annotators = np.array(annotator_labels)[sampled_annotator_idxs]
        else: sampled_annotators = np.array(annotator_labels)

        return np.mean(sampled_annotators,axis=0)
        

    def sample_our_generative_model(self, most_prob_class, ent_type="low"): 
        sampled_label = np.zeros(self.num_classes)
        if ent_type == "low": 
            sampled_top_prob = np.random.beta(self.low_ent_a, self.low_ent_b)
            # help from: https://stackoverflow.com/questions/11373192/generating-discrete-random-variables-with-specified-weights-using-scipy-or-numpy
            num_other_poss = np.random.choice(list(range(len(self.low_ent_n_poss_probs))), 1, p=self.low_ent_n_poss_probs)[0]
        else:
            sampled_top_prob = np.random.beta(self.high_ent_a, self.high_ent_b)
            num_other_poss = np.random.choice(list(range(len(self.high_ent_n_poss_probs))), 1, p=self.high_ent_n_poss_probs)[0]
        sampled_label[most_prob_class] = sampled_top_prob
        # sort the alternative labels based on semantic distance
        sorted_semantic_sims = []
        for idx in range(self.num_classes): 
            if idx != most_prob_class: sorted_semantic_sims.append((self.semantic_sims[most_prob_class, idx], idx))
        sorted_semantic_sims = sorted(sorted_semantic_sims, key=lambda x:x[0], reverse=True)
        # keep only based on num other poss
        alt_classes = [x[1] for x in sorted_semantic_sims[:num_other_poss]]
        # second most prob label based on how much is leftover and num other poss
        if num_other_poss != 0: 
            mass_leftover = 1-sampled_top_prob
            sampled_second_prob = np.random.uniform((1/num_other_poss)*mass_leftover, mass_leftover)
            sampled_label[alt_classes[0]] = sampled_second_prob
            print("second prob: ", (1/num_other_poss)*mass_leftover, mass_leftover, sampled_second_prob)

            # spread uniformly over the remaining 
            alt_classes = alt_classes[1:]
            mass_leftover = 1-np.sum(sampled_label)
            if len(alt_classes) != 0 and mass_leftover < 1.0: 
                for alt_class in alt_classes: 
                    sampled_label[alt_class] = 1/len(alt_classes) * mass_leftover

            print("final label: ", sampled_label)

        return sampled_label / np.sum(sampled_label) # ensures sums to 1 just in case

    def __getitem__(self, idx):
        '''
        Combine the imgs from cifar10 loader with human generated soft labels
        Assumes these images have transform applied (per cifar10-dataloader init)
        Returns img with target
        ''' 

        img = self.imgs[idx]

        if self.use_per_annotator: 
            # sample an annotator label and use 
            # need to convert from this new index to the "original"
            # in order to extract the matched label (todo: refactor)
            converted_idx = self.index_converter[idx]
            if converted_idx not in self.individ_annotator_labels:
                # this means it is one of the examples we don't have soft labels for, so use hard
                label = torch.tensor(self.labels[idx])
            else: 
                # sample an annotator's label! 
                individ_labels = self.individ_annotator_labels[converted_idx]
                label = torch.tensor(random.choice(individ_labels))
        else:
            # otherwise, use the agg label
            label = torch.tensor(self.labels[idx])

        # ensure all labels are in K class vector form if not already
        if len(label.shape) == 0: # index tensor 
            label = F.one_hot(label, num_classes=self.num_classes) # https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html


        # handle type conversions
        # help from: https://stackoverflow.com/questions/56741087/how-to-fix-runtimeerror-expected-object-of-scalar-type-float-but-got-scalar-typ
        label = torch.FloatTensor(label.detach().numpy().astype(np.float32))

        return (img, label)


class CIFAR10C(Dataset):
    '''
    CIFAR-10-C corrupted set
    From: https://zenodo.org/record/2535967#.YqYb4JDMLX0
    Note, this is over the cifar10 test set, which we train on b/c of cifar10h construction
    So need to extract only imgs which correspond to the 30% we holdout
    modified dataloader from example: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    '''

    def __init__(self, img_dir='~/data', cifar10c_dir="/cifar10c_data/CIFAR-10-C",
                    transform=None, split_idx_pth="./data/", split="test",
                    corruption_level=5, corruption_type="gaussian_noise"):
        self.corruption_level = corruption_level
        self.img_dir = img_dir
        # load in cifar10 test set to use the labels
        if img_dir is None: 
            # re-download cifar10 test set
            download_orig_cifar10 = True
            img_dir = "~/data" # save to
        else: download_orig_cifar10 = False
        self.img_dataset = datasets.CIFAR10(root=img_dir, train=False, download=download_orig_cifar10,
                        transform=transform) 
        data_size = len(self.img_dataset.data)

        # use imgs from cifar10c file 
        self.imgs = np.load(f"{cifar10c_dir}/{corruption_type}.npy")
        self.labels = np.load(f"{cifar10c_dir}/labels.npy")

        # according to https://zenodo.org/record/2535967#.YqYb4JDMLX0
        # and: https://github.com/hendrycks/robustness/issues/57
        # first 10000 imgs correspond to corruption level 1, then level 2, ... up to level 5 

        self.imgs = self.imgs[(corruption_level - 1)*10000:corruption_level*10000]
        self.labels = self.labels[(corruption_level - 1)*10000:corruption_level*10000]
        # if self.corruption_level == 1: 
        #     self.imgs = self.imgs[:10000]
        # else: self.imgs = self.imgs[-10000:]

        print("imgs: ", len(self.imgs), " labels: ", len(self.labels))

        self.transform = transform

        # extract names of classes
        self.classes = self.get_classes()
        # map from names of classes to indices
        self.class2idx = {class_name: idx for idx,class_name in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        # apply transform to image like in: https://github.com/jcpeterson/cifar10-human-experiments/blob/efadada2c1adc6bcdc8d86aca7e542564ff2980e/pytorch_image_classification/dataloader_c10h.py#L42
        if self.transform is not None: 
            self.imgs = [self.transform(img) for img in self.imgs]

        # optionally use specified train/test split
        # modified from Uma for implementing:
        # note: do this after getting the mixup examples, since indices are matched off the full set
        if split_idx_pth is not None:
            if split == "train": 
                indices = np.load(f"{split_idx_pth}/train_indices.npy").tolist()
            else: 
                # use test indices
                indices = np.load(f"{split_idx_pth}/test_indices.npy").tolist()

            # non-numpy filtering help from: 
            # self.imgs = [i for (i, v) in zip(list_a, filter) if v]
            self.imgs = list(np.array(self.imgs)[indices])
            self.labels = list(np.array(self.labels)[indices])

        self.imgs = self.imgs
        self.imgs=self.imgs

    def __len__(self):
        '''
        Return num data points
        '''
        return len(self.imgs) # num data points

    def get_class_per_idx(self): 
        '''
        Get semantically meaningful class labels associated with each example index
        Returns a map of {CIFAR-10/H example idx: associated label (txt), ...}
        '''
        idx2class = {v: k.capitalize() for k,v in self.img_dataset.class_to_idx.items()}
        return idx2class

    def get_classes(self): 
        '''
        Return semantically meaningful class labels 
        As an (ordered) list
        '''
        return [class_name.capitalize() for class_name in self.img_dataset.classes]

    def __getitem__(self, idx):
        '''
        Combine the imgs from cifar10 loader with human generated soft labels
        Assumes these images have transform applied (per cifar10-dataloader init)
        Returns img with target
        ''' 

        img = self.imgs[idx]
        label = torch.tensor(self.labels[idx]).to(torch.int64)

        # ensure all labels are in K class vector form if not already
        if len(label.shape) == 0: # index tensor 
            label = F.one_hot(label, num_classes=self.num_classes) # https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html

        # handle type conversions
        # help from: https://stackoverflow.com/questions/56741087/how-to-fix-runtimeerror-expected-object-of-scalar-type-float-but-got-scalar-typ
        label = torch.FloatTensor(label.detach().numpy().astype(np.float32))

        return (img, label)



class CIFAR10Sub(Dataset):
    '''
    CIFAR-10 where we can subsample and optional impoversh certain categories
    modified dataloader from example: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    '''

    def __init__(self, img_dir='~/data', cifar10c_dir="/cifar10c_data/CIFAR-10-C",
                    transform=None, split="train",
                    corruption_level=5, corruption_type="glass_blur",
                    keep_classes=set(), corrupt_classes=set(), split_seed=7):


        if os.path.exists(f"{img_dir}cifar-10-batches-py"): 
            download_orig_cifar10=False
        else: download_orig_cifar10=True

        if split == "train" or split == "val": 
            self.img_dataset = datasets.CIFAR10(root=img_dir, train=True, download=download_orig_cifar10,
                        transform=transform) 

            random.seed(split_seed)
            np.random.seed(split_seed)
            torch.manual_seed(split_seed)
            torch.cuda.manual_seed(split_seed)
            data_size = len(self.img_dataset)
            val_prop = 0.1
            val_size = int(data_size * val_prop) 
            train_size = data_size - val_size  

            all_indices = set(range(data_size))
            train_indices = set(np.random.choice(all_indices, train_size, replace = False))
            val_indices = all_indices.difference(train_indices)

            if split == "train":
                keep_indices = train_indices
            else: keep_indices = val_indices


            self.orig_imgs = [self.img_dataset[i][0] for i in range(data_size) if i in keep_indices]
            self.orig_labels = [self.img_dataset[i][1] for i in range(data_size) if i in keep_indices]

            print("num imgs: ", len(self.orig_imgs))

        else: 
            self.img_dataset = datasets.CIFAR10(root=img_dir, train=False, download=download_orig_cifar10,
                        transform=transform) 
            datasize = len(self.img_dataset)
            self.orig_imgs = [self.img_dataset[i][0] for i in range(data_size)]
            self.orig_labels = [self.img_dataset[i][1] for i in range(data_size)]

        self.origClasses = self.img_dataset.classes

        self.corruption_level = corruption_level
        self.img_dir = img_dir

        # use imgs from cifar10c file 
        self.corrupted_imgs = np.load(f"{cifar10c_dir}/{corruption_type}.npy")
        self.corrupted_assoc_labels = np.load(f"{cifar10c_dir}/labels.npy")
        # according to https://zenodo.org/record/2535967#.YqYb4JDMLX0
        # and: https://github.com/hendrycks/robustness/issues/57
        # first 10000 imgs correspond to corruption level 1, then level 2, ... up to level 5 
        self.corrupted_imgs = self.corrupted_imgs[(corruption_level - 1)*10000:corruption_level*10000]
        self.corrupted_assoc_labels = self.corrupted_assoc_labels[(corruption_level - 1)*10000:corruption_level*10000]

        self.transform = transform

        # extract names of classes
        self.orig_classes = self.get_orig_classes()
        # map from names of classes to indices
        self.origClass2idx = {class_name: idx for idx,class_name in enumerate(self.orig_classes)}
        self.origIdx2class = {idx: class_name for idx,class_name in enumerate(self.orig_classes)}

        self.class2idx = {class_name: idx for idx,class_name in enumerate(self.classes)}

        self.classes = sorted(list(keep_classes))
        self.num_classes = len(self.classes)


        keep_imgs = []
        keep_labels = []

        for idx, (img, label) in enumerate(self.orig_imgs, self.orig_labels): 
            label_txt = self.origIdx2class[label]
            if label_txt in keep_classes: 
                if split == "test" and label_txt in corrupt_classes: 
                    # store the corrupted version of the image instead
                    corrupted_obs = self.corrupted_imgs[idx]
                    keep_imgs.append(corrupted_obs)
                else: 
                    keep_imgs.append(img)

                # convert label to new idx here  
                # idx into the sorted kept classes
                new_label_idx = self.class2idx[label_txt]
                keep_labels.append(new_label_idx)

        print("Num keep: ", len(keep_imgs), len(keep_labels))

        self.imgs = keep_imgs
        self.labels= keep_labels

    def __len__(self):
        '''
        Return num data points
        '''
        return len(self.imgs) # num data points

    def get_class_per_idx(self): 
        '''
        Get semantically meaningful class labels associated with each example index
        Returns a map of {CIFAR-10/H example idx: associated label (txt), ...}
        '''
        idx2class = {idx: class_name for idx, class_name in self.classes}
        return idx2class

    def get_orig_classes(self): 
        '''
        Return semantically meaningful class labels 
        As an (ordered) list
        '''
        return [class_name.capitalize() for class_name in self.img_dataset]

    def get_classes(self): 
        '''
        Return semantically meaningful class labels 
        As an (ordered) list
        '''
        return self.classes

    def __getitem__(self, idx):
        '''
        Combine the imgs from cifar10 loader with human generated soft labels
        Assumes these images have transform applied (per cifar10-dataloader init)
        Returns img with target
        ''' 

        img = self.imgs[idx]
        label = torch.tensor(self.labels[idx]).to(torch.int64)

        # ensure all labels are in K class vector form if not already
        if len(label.shape) == 0: # index tensor 
            label = F.one_hot(label, num_classes=self.num_classes) # https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html

        # handle type conversions
        # help from: https://stackoverflow.com/questions/56741087/how-to-fix-runtimeerror-expected-object-of-scalar-type-float-but-got-scalar-typ
        label = torch.FloatTensor(label.detach().numpy().astype(np.float32))

        return (img, label)
