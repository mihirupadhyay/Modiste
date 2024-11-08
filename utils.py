import numpy as np
import torchvision.transforms as transforms
import torchvision
import torch
import sqlite3
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse
import random

import datetime
from functools import partial
from scipy.stats import entropy

from scipy.special import softmax

import pandas as pd
from sklearn.manifold import TSNE

from json import JSONEncoder
import ast

import random
from sklearn.neighbors import KDTree


def loss(y_t, opt_y_t):
    if y_t == opt_y_t:
        return 0
    else:
        return 1


def init(d, n_arms=3):
    A = []
    B = []
    for _ in range(n_arms):
        A.append(torch.eye(d))
        B.append(torch.zeros(d).unsqueeze(1))
    return A, B


def arm_selection(p_a, cost_list, gamma):
    val = gamma*(-p_a) + (1-gamma)*cost_list
    return np.argmin(val)


def compute_arm_inference_knn(z_t, all_points, results, K, arm_list, eta_list, lmbda):

    if len(all_points) < K:
        arm_select = random.choice(arm_list)
        return arm_select, arm_list.index(arm_select)

    n_arms = len(arm_list)
    all_points = np.array(all_points)
    kdt = KDTree(all_points, leaf_size=30, metric='euclidean')
    dist, ind = kdt.query(z_t.reshape(1, -1), k=K)
    temp = {}

    #estimate p_A using neighbors
    for arm in arm_list:
        temp[arm] = []

    to_debug = []
    for i in range(K):
        res = results[ind[0][i]]
        temp[res[2]].append(1-loss(res[0], res[1]))
        to_debug.append(1-loss(res[0], res[1]))
        to_debug.append(res[2])

    arm_scores = []

    for i in range(len(arm_list)):
        arm = arm_list[i]
        if len(temp[arm]) == 0:
            arm_scores.append(0)  # is this correct?
        else:
            #Compute convex combo of empirical distribution + cost
            performance = sum(temp[arm])/len(temp[arm])
            #print(performance, arm)
            combo = lmbda*performance+(1-lmbda)*(1-eta_list[i])
            #print(combo)
            arm_scores.append(combo)

    #Select arm that has highest
    arm_select = arm_list[arm_scores.index(max(arm_scores))]

    return arm_select, arm_scores.index(max(arm_scores))


def compute_arm_inference(beta, t, z_t, arm_list, eta_list, gamma):

    n_arms = len(arm_list)

    p = np.empty(shape=(n_arms))
    for a in range(n_arms):
        B_a = beta[t, a]
        p[a] = (B_a.T).dot(z_t)

    chosen_arm = arm_selection(p, eta_list, gamma)

    return arm_list[chosen_arm], chosen_arm


def compute_class_prob(beta, t, select_class, arm_list, eta_list, gamma, examples_in_class,  n_points=100):
    n_arms = len(arm_list)

    sampled_idxs = np.random.choice(
        list(range(len(examples_in_class))), n_points)
    sampled_examples = np.array(examples_in_class)[sampled_idxs]

    #compute arm for each point
    arms_pulled = [compute_arm_inference(
        beta, t, point, arm_list, eta_list, gamma)[0] for point in sampled_examples]
    arms_pulled = np.array(arms_pulled)

    # print("ARMS PULLED: ", arms_pulled)

    #compute average over arms
    temp = np.empty(shape=(n_arms))
    for i, arm in enumerate(arm_list):
        # print("ARM: ", arm, " ARMS_PULLED SHAPE: ", arms_pulled.shape)
        temp[i] = sum(arms_pulled == arm)/n_points

    return temp


def compute_prob_t(beta, t, arm_list, eta_list, gamma, labels, examples,  trial_examples=None, n_points_per_class=100):
    n_arms = len(arm_list)
    n_classes = len(labels)
    temp = np.empty(shape=(n_classes, n_arms))

    # sample a set of points from the allowed classes
    # and ensure they were not shown to the user
    poss_val_examples = {class_name: [] for class_name in set(labels)}
    kept_examples = []
    for example_id, example_data in examples.items():
        class_name = example_data["y_t"]
        if example_id not in set(trial_examples) and class_name in set(labels):
            # np.append(example_data["z_t"], 1)
            z_t = np.array(example_data["z_t"])
            poss_val_examples[class_name].append(z_t)
            kept_examples.append(example_id)

    for i, cls in enumerate(sorted(labels)):
        temp[i] = compute_class_prob(
            beta, t, cls, arm_list, eta_list, gamma, poss_val_examples[cls],  n_points_per_class)

    # check from: https://stackoverflow.com/questions/3170055/test-if-lists-share-any-items-in-python
    print("KEPT EXAMPLES!!!", bool(
        set(kept_examples) & set(trial_examples)))

    # with open("kept_examples_compute_prob.txt", "w") as f:
    #     f.write(", ".join([str(x) for x in kept_examples]))
    #     f.write("\n\n\n")
    #     f.write(", ".join([str(x) for x in trial_examples]))

    return temp


def load_model(ckpt_pth, device="cpu"):
    '''
    Load model saved at "ckpt_pth"
    '''

    checkpoint = torch.load(ckpt_pth, map_location=torch.device(device))
    model = checkpoint['net']
    if device == "cpu":
        # cpu conversion help from:
        # https://stackoverflow.com/questions/68551032/is-there-a-way-to-use-torch-nn-dataparallel-with-cpu
        model = model.module.to("cpu")
    return model


def get_dist(z):
    '''
    Convert unnormalized vec (z) to prob dist
    Softmax transform
    '''
    if torch.is_tensor(z):
        z = z.detach().cpu().numpy()

    return np.exp(z) / np.sum(np.exp(z))


def run_inference(model, x, device="cpu"):
    '''
    Run model forwards on an input
    Return feature rep, final predictive dist over K labels, and the most prob discrete label
    '''
    # run model forwards on an input
    # return feature rep, final predictive dist over K labels, and the most prob discrete label

    x = x.to(device)
    yhat_pred, feature_rep = model(x)
    yhat = torch.argmax(yhat_pred, axis=1)  # get most prob category
    # convert to valid prob dist, b/c pred isn't yet a norm'd prob dist
    yhat_dist = get_dist(yhat_pred)

    return feature_rep, yhat, yhat_dist


def sample_friend(friend_data, example_idx):
    '''
    Sample an individual annotator
    Use counts per CIFAR-10H example
    '''
    # extract individual annotator preds from the counts
    individ_annotator_preds = []
    for pred_class, n_annotators in enumerate(friend_data[example_idx]):
        individ_annotator_preds.extend(
            [pred_class for _ in range(n_annotators)])
    friend_pred = np.random.choice(individ_annotator_preds)
    return friend_pred


def run_tsne(latents, category_ids, output_path="./tsne.csv"):
    '''
    Run t-SNE over latents
    Category IDs are class label idxs
    Can optionally specify an output path for the embeddings
    '''
    # inspired by: https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-\
    #    and-t-sne-in-python-8ef87e7915b
    # compute embeddings
    print("starting tsne.....")
    embeddings = TSNE(n_jobs=4, random_state=7).fit_transform(latents)
    print("done with tsne")

    emb_df = pd.DataFrame(
        data={"emb1": embeddings[:, 0],
              "emb2": embeddings[:, 1], "id": category_ids}
    )
    emb_df.to_csv(output_path)
    return embeddings


class NumpyJSONEncoder(json.JSONEncoder):
    '''
    Allows for us to save numpy data in a json
    Code from: https://pynative.com/python-serialize-numpy-ndarray-into-json/
    '''

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.numpy().tolist()
        else:
            return super(NumpyJSONEncoder, self).default(obj)


def query_model(model, input, get_latent=False):
    pred_vec, latent_vec = model(input)
    return pred_vec, latent_vec


def dist2msg(dist):
    '''
    Convert a prob dist from a vector to a form ammenable as a server msg
    '''
    msg = "_".join([str(x) for x in dist])
    return msg


# def compute_class_prob(beta, t, class_latents, arm_list):
#     n_arms = len(arm_list)
#     #sample points
#     n_points = 100
#     # points = []
#     # for i in range(n_points):
#     #     #this is the toy 2D version (for now)
#     #     z_t = gen_point(select_class)
#     #     points.append(z_t)
#
#     print("class latents: ", class_latents)
#
#     poss_idxs = list(range(len(class_latents)))
#
#     sampled_idxs = np.random.choice(poss_idxs, n_points, replace=False)
#     points = np.array(class_latents)[sampled_idxs]
#
#     # need to add dim of [..., 1] at end
#     points = [np.append(z, 1) for z in points]
#
#     #compute arm for each point
#     arms_pulled = [compute_arm_inference(
#         beta, t, point, arm_list)[0] for point in points]
#     arms_pulled = np.array(arms_pulled)
#
#     #compute average over arms
#     temp = np.empty(shape=(n_arms))
#     for i, arm in enumerate(arm_list):
#         temp[i] = sum(arms_pulled == arm)/n_points
#
#     return temp


# def compute_prob_t(beta, t, arm_list, labels, latents_per_class):
#     n_arms = len(arm_list)
#     n_classes = len(labels)
#     temp = np.empty(shape=(n_classes, n_arms))
#     for i, cls in enumerate(labels):
#         class_latents = latents_per_class[cls]
#         temp[i] = compute_class_prob(beta, t, class_latents, arm_list)
#
#     return temp


def get_cifar10_dataset(use_trainset=False, shuffle=False, batch_size=1):
    '''
    Load in cifar-10 help from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
        ])

    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=use_trainset, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=2)
    return dataloader


def clean_user_df(user_df):
    '''
    Clean Pavlovia-saved df
    S.t. we have one entry per trial, rather than info split across several rows
    Also remove the instruction data, as well as final comments
    '''
    # find where we preload -- this indicates experiment is about to start in full
    preload_page_idx = np.where(user_df.trial_type == "preload")[0]

    if len(np.where(user_df.trial_type == "preload")[0]) != 0:
        starter_page_idx = preload_page_idx[0]
    else:
        starter_page_idx = np.where(
            user_df.trial_type == "instructions")[0][-1]

    user_df = user_df.iloc[starter_page_idx + 1:-1]  # remove comments
    # construct new data frame with reduced num entries

    # these are automatically in the first entry
    general_attributes = ["subject", "condition", "filename", "main_variant",
                          "img_id", "label", "human_corrupted", "variant"]  # ,"model1_pred", "model2_pred", "z_t"]

    cleaned_df_data = {attr: [] for attr in general_attributes}
    cleaned_df_data.update({"interfaceType": [], "modelPred": [],
                            "response": [], "humanCorrect": [], "modelCorrect": []})

    # 5 rows right now per entry
    for idx in range(0, len(user_df), 5):

        entry_data = user_df.iloc[idx: idx+5]

        # these are all just saved in the first entry for this trial
        for attr in general_attributes:
            cleaned_df_data[attr].append(entry_data[attr].iloc[0])

        # now get "special cols" saved across diff rows for the trial
        cleaned_df_data["response"].append(ast.literal_eval(
            entry_data.iloc[1]["response"])["humanClassPred"])
        cleaned_df_data["modelCorrect"].append(
            entry_data.iloc[3]["modelCorrect"])
        cleaned_df_data["humanCorrect"].append(
            entry_data.iloc[3]["humanCorrect"])
        cleaned_df_data["interfaceType"].append(
            entry_data.iloc[3]["interfaceType"])
        cleaned_df_data["modelPred"].append(entry_data.iloc[3]["modelPred"])

    cleaned_user_df = pd.DataFrame(cleaned_df_data)
    return cleaned_user_df


def clean_user_df_txt(user_df):
    '''
    Clean Pavlovia-saved df
    S.t. we have one entry per trial, rather than info split across several rows
    Also remove the instruction data, as well as final comments
    '''
    # find where we start multiple choice -- this indicates experiment is about to start in full
    start_exp_page_idx = np.where(user_df.trial_type == "call-function")[0][0]

    user_df = user_df.iloc[start_exp_page_idx:-1]  # remove comments
    # construct new data frame with reduced num entries

    # these are automatically in the first entry

    general_attributes = ["subject", "condition", "question", "main_variant", "variant",
                          "example_id", "label", "options", "llm_answer", "topic", "prompt"]

    cleaned_df_data = {attr: [] for attr in general_attributes}
    cleaned_df_data.update({"interfaceType": [], "modelPred": [],
                            "response": [], "humanCorrect": [], "modelCorrect": [], "totTime": []})

    # 5 rows right now per entry
    for idx in range(0, len(user_df), 5):

        entry_data = user_df.iloc[idx: idx+5]

        # these are all just saved in the first entry for this trial
        for attr in general_attributes:
            cleaned_df_data[attr].append(entry_data[attr].iloc[0])

        # now get "special cols" saved across diff rows for the trial
        # handle earlier keys
        human_response = ast.literal_eval(
            entry_data.iloc[1]["response"])
        if "mcAnswer" in human_response:
            human_response = human_response["mcAnswer"]
        else:
            human_response = human_response["humanClassPred"]
        cleaned_df_data["response"].append(human_response)
        cleaned_df_data["modelCorrect"].append(
            entry_data.iloc[3]["modelCorrect"])
        cleaned_df_data["humanCorrect"].append(
            entry_data.iloc[3]["humanCorrect"])
        cleaned_df_data["interfaceType"].append(
            entry_data.iloc[3]["interfaceType"])
        cleaned_df_data["modelPred"].append(entry_data.iloc[3]["modelPred"])

        # get time spent on task -- note, only includes time where human could have been interacting
        cleaned_df_data["totTime"].append(np.sum(entry_data.rt)/(1000 * 60))

    cleaned_user_df = pd.DataFrame(cleaned_df_data)
    return cleaned_user_df
