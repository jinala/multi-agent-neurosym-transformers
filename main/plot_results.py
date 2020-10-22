import os
import argparse
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.pyplot import gca
import json
import pickle
import pylab

from envs.formation_flying import *
import main.utils as utils
from main.policy import neural_policy_bptt
from tasks.task import *
from main.policy.program_attn_policy import *
from main.policy import program_attn_policy

program_attn_policy.dev = torch.device("cpu")

def get_args():
    parser = argparse.ArgumentParser(description='Plot script')
    parser.add_argument('--result_dir', type=str, default='test', help='folder to store results')
    parser.add_argument('--feature', type=str, default='orig',
        help='Program features. Can be orig, quadratic, goal, both')
    parser.add_argument('--suffix', type=str, default='',
        help='Suffix for saved program_policy')
    parser.add_argument('--env', type=str, default="FormationTorch-v0")

    return parser.parse_args()


def update_args_using_saved(args, saved_args):
    args.policy = saved_args.policy
    args.collision_dist = saved_args.collision_dist
    args.collision_penalty = saved_args.collision_penalty
    args.transformer_model_dim = saved_args.transformer_model_dim
    #args.gap = saved_args.gap
    args.col_loss = saved_args.col_loss
    #args.reward = saved_args.reward
    args.clip_speed = saved_args.clip_speed

    args.sep_key_val = saved_args.sep_key_val
    args.hops = saved_args.hops
    args.dropout_model = saved_args.dropout_model
    args.dropout_percent = saved_args.dropout_percent

    if hasattr(saved_args, 'self_attn'):
        args.self_attn = saved_args.self_attn
        args.dist_based_hard = saved_args.dist_based_hard
        args.comm_deg = saved_args.comm_deg
        args.only_obs_edge = saved_args.only_obs_edge

    if hasattr(saved_args, 'sep_self_edge'):
        args.sep_self_edge = saved_args.sep_self_edge

    if hasattr(saved_args, 'grid_l'):
        args.grid_l = saved_args.grid_l

    if hasattr(saved_args, 'noise_scale'):
        args.noise_scale = saved_args.noise_scale
    else:
        args.noise_scale = 0.0

    args.comm_deg = saved_args.comm_deg

    args.num_agents = saved_args.num_agents
    args.num_groups = saved_args.num_groups
    args.task = saved_args.task

    return args

def main():
    # Set up parameters
    args = get_args()

    results_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', args.result_dir)

    plots_dir = os.path.join(results_dir, "plots")
    utils.create_dir1(plots_dir)

    all_stats = {} # baselines -> settings -> stats

    all_settings = set()

    for subdir, dirs, files in os.walk(results_dir):
        for file in files:
            if (".json" in file) and (not "._" in file):
                x = subdir.split("/")
                if "model" in x[-2]:
                    setting = "0"
                    baseline = x[-1]
                else:
                    setting = x[-1]
                    baseline = x[-2]
                if True:
                    all_settings.add(setting)

                    with open(os.path.join(subdir, file)) as json_file:
                        stats = json.load(json_file)

                    if baseline not in all_stats:
                        all_stats[baseline] = {}

                    all_stats[baseline][setting] = stats

    results_dir_dist = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', args.result_dir+"_disthard")
    if os.path.isdir(results_dir_dist):
        for subdir, dirs, files in os.walk(results_dir_dist):
            for file in files:
                if ".json" in file and (not "._" in file):
                    x = subdir.split("/")
                    if "model" in x[-2]:
                        setting = "0"
                        baseline = x[-1]
                    else:
                        setting = x[-1]
                        baseline = x[-2]
                    if True: 
                        all_settings.add(setting)

                        with open(os.path.join(subdir, file)) as json_file:
                            stats = json.load(json_file)

                        if baseline not in all_stats:
                            all_stats[baseline] = {}

                        all_stats[baseline][setting] = stats

    desired_metrics = ['avg_cum_loss', 'max_deg_in', 'max_deg_out']
    num_baselines = len(all_stats)
    keys = sorted(list(all_stats.keys()))
    baselines = ["_".join(x.split("_")[1:]) for x in keys]
    print(baselines)

    X = np.arange(num_baselines) * 3
    stats = {}
    for metric in desired_metrics:
        # combine all settings
        Y = []
        err = []
        for baseline in keys:
            y = 0
            e = 0
            for setting in all_settings:
                y += all_stats[baseline][setting][metric + "_mean"]
                e += all_stats[baseline][setting][metric + "_std"]**2
            y = y/float(len(all_settings))
            e = (e/float(len(all_settings)))**0.5
            if metric == "avg_cum_loss" and y < 0.0:
                Y.append(10 + y)
            else:
                Y.append(y)
            err.append(e)

        stats[metric] = {
            'Y' : Y,
            'err' : err,
        }
    plt.figure(figsize=(len(baselines), 4))

    fig, ax1 = plt.subplots()
    bar1 = ax1.bar(X-0.5, stats['avg_cum_loss']['Y'], yerr=stats['avg_cum_loss']['err'], ecolor='m', capsize=2.0, width=0.5, color='y', label='avg_cum_loss')
    ax1.set_ylabel('Loss')

    ax2 = ax1.twinx()
    bar2 = ax2.bar(X[1:], stats['max_deg_in']['Y'][1:], yerr=stats['max_deg_in']['err'][1:], ecolor='m', capsize=2.0, width=0.5, label='max_deg_in')

    bar3 = ax2.bar(X[1:]+0.5, stats['max_deg_out']['Y'][1:], yerr=stats['max_deg_out']['err'][1:], ecolor='m', capsize=2.0, width=0.5, color='tab:gray', label='max_deg_out')
    ax2.set_ylabel('Degree')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, fontsize=10)

    #plt.figure(figsize=(6,4))
    #plt.title(metric)
    baseline_names = []
    for i in baselines:
        if i=='tf':
            baseline_names.append('tf-full')
        else:
            baseline_names.append(i)

    #baselines = ['tf-full' if i=='tf' else i for i in baselines]
    #plt.xticks(X, baseline_names, rotation=40)
    ax1.set_xticks(X)
    ax1.set_xticklabels(baseline_names)
    ax1.xaxis.set_tick_params(rotation=20)
    if "unlabeled" in results_dir:
        ax1.set_ylim((0, 5))
    plt.tight_layout()
    if "unlabeled" in results_dir:
        plt.title('Unlabeled goals')
        plt.savefig(plots_dir + "/unlabeled-stats.pdf")
    elif "cross" in results_dir:
        plt.title('Random cross')
        plt.savefig(plots_dir + "/random-cross-stats.pdf")
    else:
        plt.title('Random grid')
        plt.savefig(plots_dir + "/random-grid-stats.pdf")


    plt.close()





if __name__ == '__main__':
    try:
        plt.rcParams.update({'font.size': 12})
        plt.rcParams.update({'figure.autolayout': True})
        
        main()
        
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
