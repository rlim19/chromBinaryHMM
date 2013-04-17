#! /usr/bin/env python
# -*- coding: utf-8 -*-

###########################
# emission : binary (0/1) #
###########################

import numpy as np
import math
import os


def forward_backward(o, z, q, e, a, elem_size=np.longdouble):
   """
   o : observed data
   z : no.hidden states
   q : transition probs
   e : emission prob
   a : initial probs

   """

   # forward part
   alpha = np.zeros((z, len(o)), dtype=elem_size)

   for i, x_i in enumerate(o):
      for state in range(0,z):
         if i == 0:
            prev_fsum = a[state]
         else:
            prev_fsum = sum(alpha[k,i-1]*q[k,state] for k in range(0,z))
         alpha[state,i] = e[state, x_i] * prev_fsum

   prob_fw = sum(alpha[:,-1])
   print "prob_fw:" + str(prob_fw)
   print "Alpha Matrix:"
   print alpha


   #import pdb; pdb.set_trace()

   # backward part
   beta = np.zeros((z, len(o)),dtype=elem_size)
   for s in xrange(z):
      beta[s, len(o)-1] = 1
   for t in xrange(len(o)-2,-1,-1):
      for i in xrange(z):
         for j in xrange(z):
            beta[i, t] += q[i,j] * e[j][o[t+1]] * beta[j,t+1]

   # double check, the prob of bw and fw should be the same
   prob_bw = sum(a[state] * e[state, o[0]] * beta[state,0]for state in range(0,z))
   print "prob_bw:" + str(prob_bw[0])
   print "Beta Matrix:"
   print beta

   # forward/backward
   phi = (alpha * beta)/prob_fw
   print "Phi Matrix:"
   print phi


if __name__== '__main__':

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

   for_mat = forward_backward(o, 2, q, e, a)
