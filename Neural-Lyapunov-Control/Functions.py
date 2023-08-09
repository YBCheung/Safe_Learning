# -*- coding: utf-8 -*-
# from dreal import *
import torch 
import numpy as np
import random


def CheckLyapunov(x, f, V, L_V, ball_lb, ball_ub, epsilon):    
    # Given a dynamical system dx/dt = f(x,u) and candidate Lyapunov function V
    # Check the Lyapunov conditions within a domain around the origin (ball_lb ≤ sqrt(∑xᵢ²) ≤ ball_ub). 
    # If it return unsat, then there is no state violating the conditions. 

    ball = np.square(x[:,0]) + np.square(x[:,1])
    lie_derivative_of_V = L_V
    
    # ball_in_bound = logical_and(ball_lb*ball_lb <= ball, ball <= ball_ub*ball_ub)
    # ball_in_bound = ball_lb*ball_lb <= ball and ball <= ball_ub*ball_ub
    
    # Constraint: x ∈ Ball → (V(c, x) > 0 ∧ Lie derivative of V <= 0)
    ball_in_bound = np.logical_and(ball_lb*ball_lb <= ball , ball <= ball_ub*ball_ub)
    V_in_bound = np.logical_and(V>=0 , lie_derivative_of_V <= epsilon)
    condition = np.logical_and(ball_in_bound, np.logical_not(V_in_bound))
    return x[condition]

    # condition = logical_and(logical_imply(ball_in_bound, V >= 0),
    #                        logical_imply(ball_in_bound, lie_derivative_of_V <= epsilon))
    # return CheckSatisfiability(logical_not(condition),config)

def AddCounterexamples(x,CE,N): 
    # Adding CE back to sample set
    x = torch.cat((x, CE), 0)
    return x


    # # Adding CE back to sample set
    # c = []
    # nearby= []
    # for i in range(CE.size()):
    #     c.append(CE[i].mid())
    #     lb = CE[i].lb()
    #     ub = CE[i].ub()
    #     nearby_ = np.random.uniform(lb,ub,N)
    #     nearby.append(nearby_)
    # for i in range(N):
    #     n_pt = []
    #     for j in range(x.shape[1]):
    #         n_pt.append(nearby[j][i])             
    #     x = torch.cat((x, torch.tensor([n_pt])), 0)
    # return x
  
def dtanh(s):
    # Derivative of activation
    return 1.0 - s**2

def Tune(x):
    # Circle function values
    y = []
    for r in range(0,len(x)):
        v = 0 
        for j in range(x.shape[1]):
            v += x[r][j]**2
        f = [torch.sqrt(v)]
        y.append(f)
    y = torch.tensor(y)
    return y