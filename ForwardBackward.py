#! /usr/bin/env python
# -*- coding: utf-8 -*-

###########################
# emission : binary (0/1) #
###########################

import unittest
import numpy as np
import math
import os
import numpy.testing as npt


def forward_backward(o, z, q, e, a, elem_size=np.longdouble):
   """
   o : observed data
   z : no.hidden states
   q : transition probs
   e : emission prob
   a : initial probs

   """

   #import pdb; pdb.set_trace()
   # forward part
   alpha = np.zeros((z, len(o)), dtype=elem_size)

   scale_const = list()
   # iterate by columns
   for i, x_i in enumerate(o):

      for state in range(0,z):
         if i == 0:
            prev_fsum = a[state]
         else:
            prev_fsum = sum(alpha[k,i-1]*q[k,state] for k in range(0,z))
         alpha[state,i] = (e[state, x_i] * prev_fsum)

      # normalize alpha by rowsums -> with a total alpha for each time: 1.0
      # normalization (by division) to avoid overflow due to multiplication of small numbers!
      statesum = sum([alpha[state,i] for state in range(z)])
      for state in range(z):
         alpha[state,i] = alpha[state,i] / statesum
      scale_const.append(statesum)

   prob_fw = sum(alpha[:,-1])

   print "prob_fw:" + str(prob_fw)
   print "Alpha Matrix:"
   print alpha


   # backward part
   beta = np.zeros((z, len(o)),dtype=elem_size)
   for s in range(z):
      beta[s, len(o)-1] = 1
   for t in range(len(o)-2,-1,-1):
      for i in range(z):
         for j in range(z):
            beta[i, t] += q[i,j] * e[j][o[t+1]] * beta[j,t+1]

      for state in range(z):
         # normalize with scaling constant from forward mat (alpha)
         beta[state, t] = beta[state, t]/scale_const[t+1]

   # double check, the prob of bw and fw should be the same
   prob_bw = sum(a[state] * e[state, o[0]] * beta[state,0]for state in range(0,z))
   print "prob_bw:" + str(prob_bw[0])
   print "Beta Matrix:"
   print beta

   # forward/backward
   phi = (alpha * beta)
   print "Phi Matrix:"
   print phi
   return phi

class TestFwdBwd(unittest.TestCase):
   def test_fwdbw(self):
      # for test purpose (Toy example)
      # get the obs.data, returned as np array objects
      o = np.genfromtxt("profile2.test", names=True, delimiter="\t")
      o = o['001']

      # random_transition  = np.random.uniform(0,1, size = state*state).reshape(state,state)
      # e.g transition, row and col index start with 0
      q = np.array([[0.8,0.2], [0.5,0.5]])

      # random_emission = np.random.uniform(0,1, size = 2*state).reshape(state,2)
      e = np.array([[0.2,0.8], [0.7,0.3]])  #binary emission

      # random init_prob = np.random.uniform(0,1,size=1*state).reshape(state,1)
      a = np.array([[0.5],[0.5]])

      phi_mat = forward_backward(o, 2, q, e, a)
      target_phi = np.array([[0.65753425, 0.48097412, 0.81722114], [0.34246575, 0.51902588, 0.18277886]])
      prob_fw = 0.11497500000000001373
      # compare two numpy arrays
      npt.assert_almost_equal(phi_mat, target_phi, decimal = 7)


if __name__== '__main__':
   unittest.main()
