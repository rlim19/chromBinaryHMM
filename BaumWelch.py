#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from ForwardBackward import ForwardBackward
import unittest
import numpy.testing as npt

def getEmmMat(z):
   """
   z: number of states
   """
   #random_emission = np.random.uniform(0,1, size = 2*z).reshape(z,2)
   pass

def getTransMat():
   pass


def emmEstimate( obs, phi_matrix ):
   """
   obs        : observed seq (multidimensions)
   phi_matrix : matrix from forward_backward algorithm

   Estimate trans and emm
   trans: transition prob
   emm  : emission prob
   """

   #import pdb; pdb.set_trace()
   obs_matrix = np.matrix(zip(*obs))
   # estimating the emission (weigthed emission) 
   emm_estimated = (np.matrix(obs_matrix) * np.matrix(phi_matrix.T)).T

   # normalize the matrix
   # sum_phi is the sum over all the states given the whole obs.sequence
   sum_phi = np.matrix(phi_matrix).sum(axis=1)
   emm_estimated = emm_estimated/sum_phi

   return emm_estimated

def transEstimate(alpha, beta, trans, emm, no_states, obs,  elem_size = np.longdouble):
   """

   """

   #import pdb; pdb.set_trace()

   # count the number of transitions!
   new_trans = np.matrix(np.zeros((no_states, no_states), dtype=elem_size))
   for t in range(len(obs)-1):
      for k in range(no_states):
         for l in range(no_states):
            new_trans[k,l] += alpha[k, t] * trans[k,l] * emm[l, obs[t+1]] * beta[l, t+1]

   # normalizing by row to get the transition prob
   row_sum = new_trans.sum(axis=1)
   new_trans = new_trans/row_sum

   return new_trans

def transNormEstimate(alpha, beta, scale_const, trans, emm, no_states, obs,  elem_size = np.longdouble):
   """
   Using normalized alpha and beta
   """

   #import pdb; pdb.set_trace()

   # count the number of transitions
   new_trans = np.matrix(np.zeros((no_states, no_states), dtype=elem_size))
   for t in range(len(obs)-1):
      for k in range(no_states):
         for l in range(no_states):
            new_trans[k,l] += (alpha[k, t] * trans[k,l] * emm[l, obs[t+1]] * beta[l, t+1])

   # normalizing by row to get the transition prob
   row_sum = new_trans.sum(axis=1)
   new_trans = new_trans/row_sum

   return new_trans

def diffMat(mat1, mat2, threshold):
   """
   compute the difference between two matrices
   """
   #np.allclose(mat1 ,mat2, atol=threshold)
   
   pass



def hmmTrainer(obs, no_states, threshold, no_iteration,  elem_size = np.longdouble ):
   """
   EM training
   obs        : observed data
   threshold  : threshold for stopping criteria
   """

   # start probs
   a0 = np.zeros((state, 1), dtype=np.longdouble)
   a0.fill(1.0/state)
   
   # transition matrix with 0.99999 without transition 
   trans = np.zeros((state,state), dtype=elem_size)
   np.fill_diagonal(trans, 0.999999)

   # 2 : for binary (0/1)
   emm = np.random.uniform(0,1, size = 2*state).reshape(state,2)

   for i in range(no_iteration):
      # estimation
      (alpha, beta, phi, scale_const) = ForwardBackward(obs, no_states, trans, emm, a0)
   pass







class TestBaumWelch(unittest.TestCase):
   def test_emmEstimate(self):
      # for test purpose (Toy example)
      # get the obs.data, returned as np array objects
      obs = np.genfromtxt("profile.test", names=True, delimiter="\t")

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

      (alpha, beta, phi, scale_const) = ForwardBackward(obs['001'], 2, trans, emm, a0)
      emm_est = emmEstimate(obs, phi)
      target_est = np.matrix([[ 0.75406921],
                              [ 0.50297756]])
      npt.assert_almost_equal(emm_est, target_est, decimal = 7)

   def test_NonNormalizedTransEstimate(self):
      # non-normalized alpha and beta
      alpha = np.array([[0.4,0.079,0.09396], 
                        [0.15,0.1085,0.021015]])
      beta = np.array([[0.189,0.7,1], 
                       [0.2625,0.55,1]])
      trans = np.array([[0.8,0.2], 
                        [0.5,0.5]])
      emm = np.array([[0.2,0.8],
                      [0.7,0.3]])
      no_states = 2
      obs = np.genfromtxt("profile.test", names=True, delimiter="\t")
      es_trans = transEstimate( alpha, beta, trans, emm, no_states, obs['001'])
      target_est = np.matrix([[ 0.72849503, 0.27150497],
                              [ 0.54416961, 0.45583039]])

      # compare two numpy arrays
      npt.assert_almost_equal(es_trans, target_est, decimal = 7)


   def test_NormalizedTransEstimate(self):
      obs = np.genfromtxt("profile.test", names=True, delimiter="\t")
      trans = np.array([[0.8,0.2], 
                        [0.5,0.5]])
      emm = np.array([[0.2,0.8],
                      [0.7,0.3]])
      a0 = np.array([[0.5],
                     [0.5]])
      no_states = 2 
      (alpha, beta, phi, scale_const) = ForwardBackward(obs['001'], 2, trans, emm, a0)
      norm_trans = transNormEstimate(alpha, beta, scale_const, trans, emm, no_states, obs['001'])

      target_est = np.matrix([[ 0.72849503, 0.27150497],
                              [ 0.54416961, 0.45583039]])
      npt.assert_almost_equal(norm_trans, target_est, decimal = 7)


if __name__== '__main__':
   unittest.main()

