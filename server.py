'''
Server creation code
Provided from Kartik and modified for our use case
'''
#(my_new_env) mihirupadhyay@10-17-105-193 modiste-example-turing-master % curl "http://localhost:80/user123*1*C*B*0*algLinUCB_0.9"

import numpy as np
import torchvision.transforms as transforms
import torchvision
import torch
import sqlite3
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse

import datetime
from functools import partial
from scipy.stats import entropy

import random
import os
from sklearn.neighbors import KDTree
from utils import *

classes = ["A", "B", "C", "D"]

con = sqlite3.connect('shabdle.db')


class ServerHandler(BaseHTTPRequestHandler):

    def __init__(self, model, inference_data, examples_per_batch, arm_list, num_classes, n_trials, *args, **kwargs):
        print("INIT!!")

        self.model = model

        self.inference_data = inference_data
        self.examples_per_batch = examples_per_batch

        # map from "true" (y_t) label to latent
        self.latents_per_class = {}
        for _, entry_data in self.inference_data.items():
            y_t = entry_data["y_t"]
            z_t = entry_data["z_t"]
            if y_t in self.latents_per_class:
                self.latents_per_class[y_t].append(z_t)
            else:
                self.latents_per_class[y_t] = [z_t]

        self.arm_list = arm_list
        self.arm_name2idx = {arm_name: idx for idx,
                             arm_name in enumerate(self.arm_list)}

        self.n_arms = len(arm_list)

        self.dim = 2  # 3
        self.num_classes = num_classes

        self.n_trials = n_trials

        self.show_cost = 0.1
        self.eta_list = np.array([0.0, self.show_cost])
        self.alpha = 1

        # now passed in
        # self.gamma = 0.99  #  linucb
        # self.lmbda = 1.0  #  knn

        # KNN-specific params
        self.K = 8
        self.n = 25
        self.epsilon = 0.1


        self.user_data = {}

        self.t = 0
        self.prev_x = None
        self.prev_y = None

        self.labels = classes  # to make neater!!

        # Modified from:
        # BaseHTTPRequestHandler calls do_GET **inside** __init__ !!!
        # So we have to call super().__init__ after setting attributes.
        super().__init__(*args, **kwargs)

    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def do_HEAD(self):
        self.do_GET()

    def init_for_subj(self, user_id):
        # initialize parameters for a given user
        # use these to store
        A, B = init(self.dim, self.n_arms)

        beta = np.empty(shape=(self.n_trials, self.n_arms, self.dim))

        r_payoff = np.empty(self.n_trials)
        p = np.empty(shape=(self.n_trials, self.n_arms))

        arm_probabilities = np.empty(
            shape=(self.n_trials, self.num_classes, self.n_arms))

        results = {}
        all_points = []

        self.user_data[user_id] = {"A": A, "B": B, "beta": beta,
                                   "r_payoff": r_payoff,
                                   "t": 0, "prev_x": None, "prev_y": None,
                                   "last_arm": None,
                                   "p": p, "arm_probabilities": arm_probabilities,
                                   "results": results, "all_points": all_points,
                                   "lmbda": -1
                                   }

    # for each round t, call this function...
    # [IMPORTANT] make sure to append a bias term to all x values (e.g., [x1, x2,1])
    def get_next_arm(self, user_id, batch_idx, y_old_given, x_next, variant="algLinUCB_0.9"):

        lmbda = float(variant.split("_")[-1])
        print("RUNNING WITH LAMBDA: ", lmbda)

        # extract key current user params
        t = self.user_data[user_id]["t"]

        # extract examples that will be shown to this user
        #   based on their batch index
        #   this ensures we won't include the same examples in val set
        trial_examples = self.examples_per_batch[batch_idx]

        #  b/c keys were reloaded as strings
        results = {int(k): v for k, v in dict(
            self.user_data[user_id]["results"]).items()}
        # print("ALL POINTS: ", self.user_data[user_id]["all_points"], type(self.user_data[user_id]["all_points"]))
        all_points = [np.array(pt) for pt in list(
            self.user_data[user_id]["all_points"])]

        if t == 0:
            # random selection on init
            # chosen_arm = np.random.choice(
            #     list(range(len(self.arm_list))), 1)[0]
            chosen_arm = random.choice(self.arm_list)

            # TODO: check and update CIFAR!
            z_t = np.array(x_next)  # torch.Tensor(x_next)
            all_points.append(z_t)
        else:
            # note: some of these are not used with KNN alg
            x_old = torch.Tensor(self.user_data[user_id]["prev_x"])
            y_old_oracle = self.user_data[user_id]["prev_y"]
            A = torch.Tensor(self.user_data[user_id]["A"])
            B = torch.Tensor(self.user_data[user_id]["B"])
            beta = np.asarray(self.user_data[user_id]["beta"])
            last_arm = self.user_data[user_id]["last_arm"]
            r_payoff = np.asarray(self.user_data[user_id]["r_payoff"])
            p = np.asarray(self.user_data[user_id]["p"])
            arm_probabilities = np.asarray(
                self.user_data[user_id]["arm_probabilities"])

            results[t - 1] = [y_old_oracle, y_old_given, last_arm]

            print("VARIANT: ", variant, " t : ", t)

            if "LinUCB" in variant:

                # start with the update using the retrieved human performance on last trial
                # set this to the reward from the last trial

                loss_t = loss(y_old_oracle, y_old_given)

                r_payoff[t-1] = -1*loss_t  # -(eta_t + loss_t)

                last_arm = self.arm_name2idx[last_arm]

                # update parameters using previous steps' interaction datas
                A[last_arm] = A[last_arm] + np.outer(x_old, x_old.T)
                B[last_arm] = B[last_arm] + \
                    r_payoff[t-1] * x_old.unsqueeze(1)

                #pick next arm
                for a in range(self.n_arms):
                    A_inv = np.linalg.inv(A[a])
                    B_a = A_inv.dot(B[a])
                    beta[t, a] = np.squeeze(B_a)

                    z_t = torch.Tensor(x_next)
                    ucb = z_t.T.matmul(torch.Tensor(A_inv)).matmul(z_t.T)
                    p[t, a] = (B_a.T).dot(z_t) + \
                        self.alpha * np.sqrt(ucb.item())

                # chosen_arm = my_max(p[t], self.eta_list)
                chosen_arm = arm_selection(p[t], self.eta_list, lmbda)

                # compute new arm probs
                # NOTE: may do this only ever K-th trial
                arm_probabilities[t, :] = compute_prob_t(
                    beta, t, self.arm_list, self.eta_list, lmbda, classes, self.inference_data, trial_examples, n_points_per_class=50)

            elif "KNN" in variant:

                loss_t = loss(y_old_oracle, y_old_given)

                r_payoff[t-1] = -1*loss_t  # -(eta_t + loss_t)

                z_t = np.array(x_next)  # torch.Tensor(x_next)
                if t < self.n:  # pure exploration
                    chosen_arm = random.choice(self.arm_list)

                else:
                    if random.random() < self.epsilon:  # epsilon exploration
                        chosen_arm = random.choice(self.arm_list)
                    else:  # pick best arm from k nearest neighbors
                        chosen_arm, arm_idx = compute_arm_inference_knn(
                            z_t, all_points, results, self.K, self.arm_list, self.eta_list, lmbda)  # , self.inference_data, trial_examples)

                all_points.append(z_t)

            # make several updates to user params
            self.user_data[user_id]["A"] = A
            self.user_data[user_id]["B"] = B
            self.user_data[user_id]["beta"] = beta
            self.user_data[user_id]["r_payoff"] = r_payoff
            self.user_data[user_id]["p"] = p
            self.user_data[user_id]["arm_probabilities"] = arm_probabilities

        self.user_data[user_id]["results"] = results
        self.user_data[user_id]["all_points"] = all_points
        self.user_data[user_id]["lmbda"] = lmbda

        #  https://stackoverflow.com/questions/40429917/in-python-how-would-you-check-if-a-number-is-one-of-the-integer-types
        if isinstance(chosen_arm, (int, np.integer)):
            chosen_arm = self.arm_list[chosen_arm]
        last_arm = chosen_arm  # save final arm idx
        self.user_data[user_id]["last_arm"] = last_arm

        return chosen_arm

    def do_GET(self):
        post_data = self.path
        print("RECEIVED: ", post_data)

        user_id, example_idx, oracle_label, prev_human_pred_label, batch_idx, variant = post_data.split(
            "*")

        #user_save_pth = f"tmp_user_data/{user_id}_params.json"
        user_save_pth = f"/Users/mihirupadhyay/Desktop/Modiste/modiste-example-turing-master/tmp_user_data/{user_id}_params.json"


        if os.path.exists(user_save_pth):
            with open(user_save_pth, "r") as f:
                user_data = json.load(f)
            self.user_data = json.loads(user_data)
        else:
            self.init_for_subj(user_id)

        example_idx = int(example_idx)

        batch_idx = int(batch_idx)

        print("example: ", example_idx,
              prev_human_pred_label, " batch: ", batch_idx)

        input_rep = self.inference_data[example_idx]["z_t"]

        # expand dim
        if self.dim == 3:
            input_rep = np.append(np.array(input_rep), 1)
        else:
            input_rep = np.array(input_rep)

        # select action for the next interface using algs
        action = self.get_next_arm(
            user_id, batch_idx, prev_human_pred_label, input_rep, variant)

        print("action: ", action)

        self.user_data[user_id]["t"] += 1
        self.user_data[user_id]["prev_x"] = input_rep
        self.user_data[user_id]["prev_y"] = oracle_label

        print("CURRENT TIMESTEP: ", self.user_data[user_id]["t"])

        self._set_response()

        # write out updated user information
        # note, need to specially encode numpy data to ensure we can save to json format
        # help from: https://pynative.com/python-serialize-numpy-ndarray-into-json/
        encoded_user_data = json.dumps(self.user_data, cls=NumpyJSONEncoder)
        with open(user_save_pth, "w") as f:
            json.dump(encoded_user_data, f)

        self.wfile.write(
                    f'{action}*'.encode('utf-8'))


if __name__ == '__main__':

    '''
    Load in pre-processed model preds and latents
    We want to retrieve by the idx
    '''
    # model_data_pth = "6class_seperable_model_pred_data_compressed.json"
    data_pth = "server_data.json"
    with open(data_pth, "r") as f:
        server_data = json.load(f)
    # contains maps from the original cifar example idx to z_t and "gt" y_t
    # as well as the examples in any given batch
    emb_data = {int(k): v for k, v in server_data["emb_data"].items()}
    examples_per_batch = np.array(server_data["examples_per_batch"])

    arm_list = [
         "defer", "showPred"] 
    num_classes = len(
         classes)
    n_trials = len(examples_per_batch[0])
    net = None  #  placeholder for if we want to pass a model in the future

    # bind some arguments
    # help from: https://stackoverflow.com/questions/21631799/how-can-i-pass-parameters-to-a-requesthandler
    #handler = partial(ServerHandler, net, emb_data, examples_per_batch, arm_list,
    #                  num_classes, n_trials)

    #httpd = HTTPServer(
    #     ('0.0.0.0', 80), handler)
    #try:
    #    httpd.serve_forever()
    #except KeyboardInterrupt:
    #    pass
    #httpd.server_close()
    #con.close()
    #con.close()
    handler = partial(ServerHandler, net, emb_data, examples_per_batch, arm_list, num_classes, n_trials)

    with HTTPServer(('0.0.0.0', 80), handler) as httpd:
        try:
            print("Server started at http://0.0.0.0:80")
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer interrupted by user, stopping...")
        finally:
            httpd.server_close()
            con.close()  # Close your connection here properly
            print("Server stopped and connection closed.")

