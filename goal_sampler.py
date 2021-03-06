from collections import deque
import numpy as np
from utils import get_idxs_per_relation
from mpi4py import MPI
import os
import pickle
import pandas as pd
from mpi_utils import logger


class GoalSampler:
    def __init__(self, args):
        self.num_rollouts_per_mpi = args.num_rollouts_per_mpi
        self.rank = MPI.COMM_WORLD.Get_rank()

        self.goal_dim = args.env_params['goal']
        self.relation_ids = get_idxs_per_relation(n=args.n_blocks)

        self.discovered_goals = []
        self.discovered_goals_str = []

        self.init_stats()

    def sample_goal(self, n_goals, evaluation):
        """
        Sample n_goals goals to be targeted during rollouts
        evaluation controls whether or not to sample the goal uniformly or according to curriculum
        """
        if evaluation and len(self.discovered_goals) > 0:
            goals = np.random.choice(self.discovered_goals, size=self.num_rollouts_per_mpi)
            self_eval = False
        else:
            if len(self.discovered_goals) == 0:
                goals = np.random.choice([-1., 1.], size=(n_goals, self.goal_dim))
                self_eval = False
            # if no curriculum learning
            else:
                # sample uniformly from discovered goals
                goal_ids = np.random.choice(range(len(self.discovered_goals)), size=n_goals)
                goals = np.array(self.discovered_goals)[goal_ids]
                self_eval = False
        return goals, self_eval

    def update(self, episodes, t):
        """
        Update discovered goals list from episodes
        Update list of successes and failures for LP curriculum
        Label each episode with the last ag (for buffer storage)
        """
        all_episodes = MPI.COMM_WORLD.gather(episodes, root=0)

        if self.rank == 0:
            all_episode_list = [e for eps in all_episodes for e in eps]

            for e in all_episode_list:
                # Add last achieved goal to memory if first time encountered
                if str(e['ag_binary'][-1]) not in self.discovered_goals_str:
                    self.discovered_goals.append(e['ag_binary'][-1].copy())
                    self.discovered_goals_str.append(str(e['ag_binary'][-1]))

        self.sync()

        return episodes

    def generate_eval_goals(self):
        """ Generates a set of goals for evaluation. This set comprises :
        - One relation with close == True .
        - One relation with above == True
        - Two relations with close == True in one of them
        - Two relations with close == True in both of them
        - Two relations with above == True in one and close == False in the other
        - Two relations with above == True in one and close == True in the other
        - Two relations with above == True in one and above == True in the other
        - Three whole relations for the 7 above cases"""
        if self.use_masks:
            masks = np.array([np.array([0, 1, 1, 0, 1, 0, 1, 1, 1]), np.array([0, 1, 1, 0, 1, 0, 1, 1, 1]),
                              np.array([0, 0, 1, 0, 0, 0, 1, 0, 1]), np.array([0, 0, 1, 0, 0, 0, 1, 0, 1]),
                              np.array([0, 0, 1, 0, 0, 0, 1, 0, 1]), np.array([0, 0, 1, 0, 0, 0, 1, 0, 1]),
                              np.array([0, 1, 0, 0, 1, 0, 0, 1, 0]),
                              np.zeros(9), np.zeros(9), np.zeros(9), np.zeros(9), np.zeros(9)])
        else:
            masks = np.zeros((12, 9))
        gs = np.array([np.array([1., -10., -10., -1., -10., -1., -10., -10., -10.]), np.array([1., -10., -10., 1., -10., -1., -10., -10., -10.]),

                       np.array([1., -1., -10., -1., -1., -1., -10., -1., -10.]), np.array([1., 1., -10., -1., -1., -1., -10., -1., -10.]),
                       np.array([1., -1., -10., -1., -1., 1., -10., -1., -10.]), np.array([1., 1., -10., -1., 1., -1., -10., -1., -10.]),
                       np.array([1., -10., 1., 1., -10., -1., 1., -10., -1.]),

                       np.array([1., -1., -1., -1., -1., -1., -1., -1., -1.]), np.array([1., -1., -1., 1., -1., -1., -1., -1., -1.]),

                       np.array([1., 1., -1., -1., -1., -1., -1., -1., -1.]),
                       np.array([1., 1., 1., -1., -1., 1., -1., -1., -1.]),
                       np.array([1., -1., 1., 1., -1., -1., 1., -1., -1.])
                       ])
        return gs, masks

    def sync(self):
        self.discovered_goals = MPI.COMM_WORLD.bcast(self.discovered_goals, root=0)
        self.discovered_goals_str = MPI.COMM_WORLD.bcast(self.discovered_goals_str, root=0)

    def build_batch(self, batch_size):
        goal_ids = np.random.choice(np.arange(len(self.discovered_goals)), size=batch_size)
        return goal_ids

    def init_stats(self):
        self.stats = dict()
        # Number of classes of eval
        if self.goal_dim == 30:
            n = 12
        else:
            n = 6
        for i in np.arange(1, n+1):
            self.stats['Eval_SR_{}'.format(i)] = []
            self.stats['Av_Rew_{}'.format(i)] = []
        self.stats['epoch'] = []
        self.stats['episodes'] = []
        self.stats['global_sr'] = []
        self.stats['nb_discovered'] = []
        keys = ['goal_sampler', 'rollout', 'gs_update', 'store', 'norm_update',
                'policy_train', 'eval', 'epoch', 'total']
        for k in keys:
            self.stats['t_{}'.format(k)] = []

    def save(self, epoch, episode_count, av_res, av_rew, global_sr, time_dict):
        self.stats['epoch'].append(epoch)
        self.stats['episodes'].append(episode_count)
        self.stats['global_sr'].append(global_sr)
        for k in time_dict.keys():
            self.stats['t_{}'.format(k)].append(time_dict[k])
        self.stats['nb_discovered'].append(len(self.discovered_goals))
        for g_id in np.arange(1, len(av_res) + 1):
            self.stats['Eval_SR_{}'.format(g_id)].append(av_res[g_id-1])
            self.stats['Av_Rew_{}'.format(g_id)].append(av_rew[g_id-1])
            # self.stats['#Rew_{}'.format(g_id)].append(self.rew_counters[oracle_id])
            # self.stats['#Target_{}'.format(g_id)].append(self.target_counters[oracle_id])
