"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

__author__ = "Allan Reyes"
__email__ = "reyallan@gatech.edu"
__userId__ = "arx3"

import logging
import numpy as np
import random as rand
import sys

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_states = num_states
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = rar
        self.decay = radr
        self.dyna = dyna
        self.qtable = np.zeros((num_states, num_actions))
        # Dyna-Q models of T and R
        self.t_counts = np.full((num_states, num_actions, num_states), 0.00001)
        self.r_prime = np.zeros((num_states, num_actions))

    def author(self):
        return 'arx3'

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        action = self._get_action(s)

        # Don't forget to keep track of the new state and action!
        self.s = s
        self.a = action

        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        action = self._get_action(s_prime)
        self._learn(s_prime, r)
        self._update()

        # Don't forget to keep track of the new state and action!
        self.s = s_prime
        self.a = action

        return action

    def _get_action(self, state):
        if rand.random() <= self.epsilon:
            return self._random_action(state)

        return self._optimal_action(state)

    def _random_action(self, state):
        action = rand.randint(0, self.num_actions - 1)
        return action

    def _optimal_action(self, state):
        action = np.argmax(self.qtable[state, :])
        return action

    def _learn(self, state_prime, reward):
        self._update_q_table((self.s, self.a, reward, state_prime))
        self._update_transition_model(state_prime)
        self._update_reward_model(reward)
        self._run_dyna_q()

    def _update_transition_model(self, state_prime):
        self.t_counts[self.s, self.a, state_prime] += 1

    def _update_reward_model(self, reward):
        self.r_prime[self.s, self.a] += self.alpha * \
                                        (reward - self.r_prime[self.s, self.a])

    def _run_dyna_q(self):
        if self.dyna <= 0:
            return

        # Pre-compute the transition probabilities for all s,a -> s' combinations
        # This divides all s,a pairs overs the sum of all their transitions s'
        # which effectively computes the probability of reaching each state s'
        # starting from s and taking action a
        transitions = self.t_counts / np.sum(self.t_counts, axis=0)

        for _ in range(self.dyna):
            state = rand.randint(0, self.num_states - 1)
            action = rand.randint(0, self.num_actions - 1)
            # Infer s' by using the state with the highest probability
            state_prime = np.argmax(transitions[state, action, :])
            reward = self.r_prime[state, action]

            self._update_q_table((state, action, reward, state_prime))

    def _update_q_table(self, experience):
        state, action, reward, state_prime = experience

        # Compute the temporal difference (TD) error: target - output
        target = reward + self.gamma * np.max(self.qtable[state_prime, :])
        td_error = target - self.qtable[state, action]
        self.qtable[state, action] += self.alpha * td_error

    def _update(self):
        self.epsilon *= self.decay

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
