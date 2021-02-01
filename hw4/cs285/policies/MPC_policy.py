import numpy as np

from .base_policy import BasePolicy


class MPCPolicy(BasePolicy):

    def __init__(self,
        sess,
        env,
        ac_dim,
        dyn_models,
        horizon,
        N,
        **kwargs):
        super().__init__(**kwargs)

        # init vars
        self.sess = sess
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

    def sample_action_sequences(self, num_sequences, horizon):
        # TODO(Q1) uniformly sample trajectories and return an array of
        # dimensions (num_sequences, horizon, self.ac_dim)
        return np.random.uniform(self.low,self.high,size=[num_sequences,horizon,self.ac_dim])

    def get_action(self, obs):

        if self.data_statistics is None:
            # print("WARNING: performing random actions.")
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        #sample random actions (Nxhorizon)
        candidate_action_sequences = self.sample_action_sequences(num_sequences=self.N, horizon=self.horizon)

        # a list you can use for storing the predicted reward for each candidate sequence
        predicted_rewards_per_ens = []

        # for model in self.dyn_models:
        #     predicted_rewards_per_ens.append([])
        #     for n in range(self.N):
        #         obs_for_this_seq = []
        #         obs_for_this_seq.append(obs)
        #         for i in range(self.horizon):
        #             action_now = candidate_action_sequences[n,i]
        #             obs_now = obs_for_this_seq[-1]
        #             obs_for_this_seq.append(model.get_prediction(obs_now,action_now,self.data_statistics))
        #     predicted_rewards_per_ens[-1].append(self.env.get_reward(np.array(obs_for_this_seq),candidate_action_sequences[n]))
        # for model in self.dyn_models:
        #     predicted_rewards_per_ens.append([0.0]*self.N)
        #     if len(obs.shape)>1:
        #         now_obs = obs
        #     else:
        #         now_obs = obs[None]
        #     now_obs = np.tile(obs, [self.N, 1])
        #     for i in range(self.horizon):
        #         now_action = candidate_action_sequences[:,i]
        #         predicted_rewards_per_ens[-1][:] = predicted_rewards_per_ens[-1][:] + self.env.get_reward(now_obs,now_action)
        #         now_obs = model.get_prediction(now_obs,now_action,self.data_statistics)

        for model in self.dyn_models:   
            if len(obs.shape)>1:
                cur_obs = obs
            else:
                cur_obs = obs[None]
            cur_obs = np.tile(obs, [self.N, 1])
            cur_rwds = np.zeros(self.N)
            for t in range(self.horizon):
                r_tot, dones = self.env.get_reward(cur_obs, candidate_action_sequences[:,t,:])
                cur_rwds += r_tot
                cur_obs = model.get_prediction(cur_obs, candidate_action_sequences[:,t,:], self.data_statistics)
            predicted_rewards_per_ens.append(cur_rwds)

            # TODO(Q2)

            # for each candidate action sequence, predict a sequence of
            # states for each dynamics model in your ensemble

            # once you have a sequence of predicted states from each model in your
            # ensemble, calculate the reward for each sequence using self.env.get_reward (See files in envs to see how to call this)

        # calculate mean_across_ensembles(predicted rewards).
        # the matrix dimensions should change as follows: [ens,N] --> N
        predicted_rewards = np.mean(predicted_rewards_per_ens,axis=0) # TODO(Q2)

        # pick the action sequence and return the 1st element of that sequence
        best_index = np.argmax(predicted_rewards) #TODO(Q2)
        best_action_sequence = candidate_action_sequences[best_index] #TODO(Q2)
        action_to_take = best_action_sequence[0] # TODO(Q2)
        return action_to_take[None] # the None is for matching expected dimensions
