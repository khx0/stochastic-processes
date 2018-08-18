#!/usr/bin/python
# -*- coding: utf-8 -*-
##########################################################################################
# author: Nikolas Schnellbaecher
# contact: khx0@posteo.net
# date: 2018-08-18
# file: gillespie.py
##########################################################################################

def gillespie_ssa(params, propensityFunc, updateMatrix, initialState, timePoints):
    """
    Gillespie SSA
	
	This is a fixed time point Gillespie SSA. This means the output
	will be returned on a prespecified time-grid and not be true
	event times from the internal Gillespie algorithm. This facilities
	any post-processing of the simulation trajectories, but does in fact
	discard a certain degree of information, namely the exact event times.
	
    Input parameters:
	
    params :
    	tuple of system parameters
    propensityFunc :  a function handle
        Function with signature f(params, population) that takes the current
        population of particle counts and returns an array of propensities
        for each reaction.
    updateMatrix : matrix with dimensions(shape)  (nReactions, nSpecies), 
    	i.e. nReactions x nSpecies. The [i, j] entry of the updateMatrix matrix
    	specifies the copy number change of the chemical species j
        when reaction (channel) i occurs (fires).
    initialState :
    	initial population of the chemical species
        Array of initial populations of all chemical species.
    timePoints :
    	Array of time points at which we sample the pdf which
  		solves the chemical master equation.

    Returns
    -------
    out : array of dimension (shape) (nTimePoints, nSpecies)
    	The [i, j] entry is the count of chemical species j at time
        timePoints[i].
    """
    
    # initialize output array
    out = np.empty((len(timePoints), updateMatrix.shape[1]), dtype = np.int)
    
    # initialize and perform simulation
    iterationTime = 1
    iteration = 0
    time = timePoints[0]
    currentState = initialState.copy()
    out[0, :] = currentState
   
    while iteration < len(timePoints):
    
        while time < timePoints[iterationTime]:
            
            # draw the event and time step
            event, dt = drawEvent(params, propensityFunc, currentState)

            if event == -1:
                # set the current time t to the last time point
                time = timePoints[-1]
                # update the population_previous array to the current values
                previousState = currentState.copy()
                # break the inner while loop (time-loop)
                break
        
            else:
                # update the population
                previousState = currentState.copy()
                currentState += updateMatrix[event, :]
            
                # increment time
                time += dt
        
        # update the index
        iteration = np.searchsorted(timePoints > time, True)
        
        # update the population
        out[iterationTime : min(iteration, len(timePoints))] = previousState
        
        # increment index
        iterationTime = iteration
    
    return out