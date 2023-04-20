import numpy as np
import math 

def generateUniformData(N, l, u, gVar):
	'''generateUniformData(N, l, u, gVar): Generate N uniformly spaced data points 
    in the range [l,u) with zero-mean Gaussian random noise with standard deviation gVar'''
	# x = np.random.uniform(l,u,N)
	step = (u-l)/(N)
	x = np.arange(l+step/2,u+step/2,step)
	e = np.random.normal(0,gVar,N)
	t = np.sin(2*math.pi*x) + e
	return x,t
