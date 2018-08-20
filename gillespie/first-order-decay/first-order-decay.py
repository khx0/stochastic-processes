#!/usr/bin/python
# -*- coding: utf-8 -*-
##########################################################################################
# author: Nikolas Schnellbaecher
# contact: khx0@posteo.net
# date: 2018-08-20
# file: first-order-decay.py
##########################################################################################

import sys
sys.path.append('../')
import time
import datetime
import os
import numpy as np

from gillespie import gillespie_ssa

def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

now = datetime.datetime.now()
now = "%s-%s-%s" %(now.year, str(now.month).zfill(2), str(now.day).zfill(2))

BASEDIR = os.path.dirname(os.path.abspath(__file__))
RAWDIR = os.path.join(BASEDIR, 'raw')
OUTDIR = os.path.join(BASEDIR, 'out')

ensure_dir(RAWDIR)

##########################################################################################
##########################################################################################
# Gillespie Model Specification
# First oder decay model
##########################################################################################
# specify the state update matrix updateMatrix
updateMatrix = np.array([[-1]], dtype = np.int)

def getPropensity(params, state):
    """
    Returns an array of propensities given a set of
    parameters and an array containing the current state.
    """
    # unpack parameters
    gamma = params
    
    # unpack population
    x = state
    
    return np.array([gamma * x])
##########################################################################################
##########################################################################################

if __name__ == '__main__':

    # fix random number seed for reproducibility
    np.random.seed(42)
    
    params = np.array([1.0])
    timePoints = np.linspace(0, 10.0, 101)
    
    initialState = np.array([20], dtype = int)
    
    traj = np.empty((len(timePoints), 1))
    
    # run SSA
    traj[:, :] = gillespie_ssa(params, 
                               getPropensity, 
                               updateMatrix,
                               initialState,
                               timePoints)
    
    # check trajectory shape after the SSA has run
    assert traj.shape == (len(timePoints), 1), "Error: Shape assertion failed."
    print("Trajectory shape =", traj.shape)
    
    