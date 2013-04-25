#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import numpy as np
from ForwardBackward import ForwardBackward
import unittest
import numpy.testing as npt


def emmEstimate( obs, phi_matrix, elem_size= np.longdouble ):
   """
   obs        : observed seq (multidimensions)
   phi_matrix : matrix from forward_backward algorithm

   Estimate trans and emm
   trans: transition prob
   emm  : emission prob
   """

   #import pdb; pdb.set_trace()
   #obs_matrix = np.matrix(zip(*obs))
   obs_matrix = np.matrix(obs)

   # estimating the emission (weigthed emission) for obs of 1 
   emm_estimated1 = np.matrix((np.matrix(obs_matrix) * np.matrix(phi_matrix.T)).T, dtype=elem_size)

   # normalize the matrix
   # sum_phi is the sum over all the states given the whole obs.sequence
   sum_phi = np.matrix(np.matrix(phi_matrix).sum(axis=1), dtype=elem_size)
   np.seterr(invalid='ignore')
   emm_estimated1 = np.matrix(emm_estimated1/sum_phi, dtype=elem_size)
   #emm_estimated1 = np.nan_to_num(emm_estimated1)

   # estimated emission for 0
   emm_estimated0 = np.matrix(1 - np.matrix(emm_estimated1), dtype=elem_size)
   # combine emission for 0 and 1 together
   emm_estimated = np.matrix(np.concatenate((emm_estimated0, emm_estimated1), axis = 1), dtype=elem_size)
   #emm_estimated = np.nan_to_num(emm_estimated)

   return emm_estimated

def transEstimate(alpha, beta, trans, emm, no_states, obs,  elem_size = np.longdouble):
   """
   alpha and beta are not normalized upon iteration!
   """

   #import pdb; pdb.set_trace()

   # count the number of transitions!
   new_trans = np.matrix(np.zeros((no_states, no_states), dtype=elem_size))
   for t in range(len(obs)-1):
      for k in range(no_states):
         for l in range(no_states):
            new_trans[k,l] += alpha[k, t] * trans[k,l] * emm[l, obs[t+1]] * beta[l, t+1]

   # normalizing by row to get the transition prob
   row_sum = np.matrix(new_trans.sum(axis=1), dtype = elem_size)
   new_trans = np.matrix(new_trans/row_sum, dtype=elem_size)
   #new_trans = np.nan_to_num(new_trans)

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
   row_sum = np.matrix(new_trans.sum(axis=1), dtype=elem_size)
   np.seterr(invalid='ignore')
   new_trans = np.matrix(new_trans/row_sum, dtype=elem_size)
   #new_trans = np.nan_to_num(new_trans)

   return new_trans

def hmmTrainer(obs, no_states,  no_iteration, threshold=0.01, elem_size = np.longdouble):
   """
   EM training
   obs        : observed data
   threshold  : threshold for stopping criteria

   Stopping criteria if and only if the |diff| of old vs new the emission and transition matrix < threshold
   """

   # start probs
   a0 = np.zeros((no_states, 1), dtype=elem_size)
   np.matrix(a0.fill(1.0/no_states), dtype = elem_size)

   # transition matrix with 0.9 without transition 
   trans = np.zeros((no_states, no_states), dtype=elem_size)
   np.fill_diagonal(trans, 0.9)
   # normalize the transition matrix 
   row, column = np.where(trans == 0.0)
   index_zero = zip (row, column)
   for (i,j) in index_zero:
      trans[i,j] = (1-0.9)/(no_states-1)

   #trans = np.array([[0.9,0.1], [0.1,0.9]])
   # emm matrix 2 : for binary (0/1)
   emm = np.matrix(np.random.uniform(0,1, size = 2 * no_states).reshape(no_states,2), dtype=elem_size)

   for i in range(no_iteration):
      # E-step
      (alpha, beta, phi, scale_const) = ForwardBackward(obs, no_states, trans, emm, a0, elem_size)

      # M-step
      new_trans = transNormEstimate( alpha, beta, scale_const, trans, emm, no_states, obs, elem_size)
      new_emm = emmEstimate(obs,phi)
      if (np.allclose(new_trans, trans, atol=threshold) and 
         np.allclose(new_emm, emm, atol=threshold)):
         print "Iteration: " + str(i)
         print "reach stop"
         break
      trans = new_trans
      emm = new_emm
      print i

   return (trans, emm)


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
      emm_est = emmEstimate(obs['001'], phi)
      target_est = np.matrix([[ 0.24593079,  0.75406921],
                              [ 0.49702244,  0.50297756]])
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

   def test_HMMtrainer(self):
      #import pdb; pdb.set_trace()
      #obs = np.genfromtxt("chrs.txt", names=True, delimiter="\t")
      #obs = np.genfromtxt("profile.test", names=True, delimiter="\t")
      obs = np.genfromtxt("test20.txt", names=True, delimiter="\t")
      (trans, emm) = hmmTrainer(obs['001'], 4, 10,  0.001)
      print "Trans\n %s"%(trans)
      print "Emm\n %s" %(emm)


if __name__== '__main__':
   unittest.main()
