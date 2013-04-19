#! /usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################################
# forward-backward: compute probability of being in state k, at time t given the obs.sequence  #
# emission : binary (0/1)                                                                      #
################################################################################################

import numpy as np
import math
import os
import unittest
import numpy.testing as npt

def ForwardBackward(obs, no_states, trans, emm, a0, elem_size=np.longdouble):
   """
   obs       : observed data
   no_states : no.hidden states
   trans     : transition probs
   emm       : emission prob
   a0        : initial probs

   output:
   - phi matrix, the probability of being in state k, at each t step
     given the obs.sequence

   Normalization according to :
      Inference in Hidden Markov Models (Ryden, et al.): page 123
   Normalization is required to avoid overflow upon interation of E-step
   """

   # for debugging
   #import pdb; pdb.set_trace()

   ################
   # forward part #
   ################

   # create the array containing zeros
   # rows are states
   # cols are the time step of the obs.sequence
   alpha = np.zeros((no_states, len(obs)), dtype=elem_size)

   scale_const = list()
   # iterate over the obs.sequence
   for t, x_t in enumerate(obs):

      for state in range(no_states):
         if t == 0:
            prev_fsum = a0[state]
         else:
            prev_fsum = sum(alpha[k,t-1]*trans[k,state] for k in range(no_states))
         alpha[state,t] = (emm[state, x_t] * prev_fsum)

      # normalize alpha by rowsums -> with a total alpha for each time: 1.0
      # normalization (by division) to avoid overflow due to multiplication of small numbers!
      statesum = sum([alpha[state,t] for state in range(no_states)])
      for state in range(no_states):
         alpha[state,t] = alpha[state,t] / statesum
      scale_const.append(statesum)
   alpha = np.nan_to_num(alpha)

   #prob_fw = sum(alpha[:,-1])
   #print "prob_fw:\n"
   #print prob_fw

   #################
   # backward part #
   #################
   #import pdb; pdb.set_trace()
   beta = np.zeros((no_states, len(obs)),dtype=elem_size)

   # fill backward from the last obs.sequence
   for state in range(no_states):
      #beta[state, len(obs)-1] = 1
      beta[state, len(obs)-1] = scale_const[-1]

   # t is the time step of the obs.sequence
   for t in range(len(obs)-2,-1,-1):
      # transition from i[t] to j[t+1] 
      for i in range(no_states):
         for j in range(no_states):
            beta[i, t] += (trans[i,j] * emm[j, obs[t+1]] * beta[j,t+1])/scale_const[t]


   # double check, the prob of bw and fw should be the same
   #prob_bw = sum(a0[state] * emm[state, obs[0]] * beta[state,0]for state in range(no_states))
   #print "prob_bw:\n"
   #print prob_bw

   beta = np.nan_to_num(beta)


   # forward/backward
   phi = np.matrix((alpha * beta), dtype= elem_size)
   colsum = np.matrix(np.matrix(phi).sum(axis=0), dtype = elem_size)
   phi = np.matrix(np.matrix(phi)/colsum, dtype = elem_size)
   phi = np.nan_to_num(phi)

   return (alpha, beta, phi, scale_const)

class TestFwdBwd(unittest.TestCase):
   def test_fwdbw(self):
      # for test purpose (Toy example)
      # get the obs.data, returned as np array objects
      obs = np.genfromtxt("profile2.test", names=True, delimiter="\t")
      obs = obs['001']

      # random_transition  = np.random.uniform(0,1, size = state*state).reshape(state,state)
      # e.g transition, row and col index start with 0
      trans = np.array([[0.8,0.2], 
                        [0.5,0.5]])

      # random_emission = np.random.uniform(0,1, size = 2*state).reshape(state,2)
      emm = np.array([[0.2,0.8], 
                      [0.7,0.3]])  #binary emission

      # random init_prob = np.random.uniform(0,1,size=1*state).reshape(state,1)
      a0 = np.array([[0.5],
                     [0.5]])

      (alpha, beta, phi, scale_const) = ForwardBackward(obs, 2, trans, emm, a0)
      target_phi = np.array([[0.65753425, 0.48097412, 0.81722114], 
                             [0.34246575, 0.51902588, 0.18277886]])

      # compare two numpy arrays
      npt.assert_almost_equal(phi, target_phi, decimal = 7)

if __name__== '__main__':
   unittest.main()
