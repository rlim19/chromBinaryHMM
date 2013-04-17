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


def baumWelchEstimate( obs, phi_matrix ):
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

   return emm_estimated


def hmmTrainer():
   pass

class TestBaumWelch(unittest.TestCase):
   def test_Estimate(self):
      # for test purpose (Toy example)
      # get the obs.data, returned as np array objects
      obs = np.genfromtxt("profile.test", names=True, delimiter="\t")

      # random_transition  = np.random.uniform(0,1, size = state*state).reshape(state,state)
      # e.g transition, row and col index start with 0
      trans = np.array([[0.8,0.2], [0.5,0.5]])

      # random_emission = np.random.uniform(0,1, size = 2*state).reshape(state,2)
      emm = np.array([[0.2,0.8], [0.7,0.3]])  #binary emission

      # random init_prob = np.random.uniform(0,1,size=1*state).reshape(state,1)
      a0 = np.array([[0.5],[0.5]])

      phi_mat = ForwardBackward(obs['001'], 2, trans, emm, a0)
      emm_est = baumWelchEstimate(obs, phi_mat)

      target_est = np.matrix([[ 1.47475539],[ 0.52524461]])

      # compare two numpy arrays
      npt.assert_almost_equal(emm_est, target_est, decimal = 7)

if __name__== '__main__':
   unittest.main()

