#
# Catch-all for useful little snippets that don't need organizing. 
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import random
import numpy as np
import os,sys,pickle,re
import math
import time
from math import pi as Pi
import scipy.special
import itertools
import warnings
from scipy.weave import inline
from collections import defaultdict
from collections import Counter
warnings.simplefilter(action = "ignore", category = FutureWarning)

#
# GLOBALS
#	Any global variables of the code must be put here, and must be in all caps.
#	Global variables are almost never acceptable except in these few cases
#

MAX_ATOMIC_NUMBER = 25
HAS_PYSCF = False
HAS_EMB = False
HAS_TF = False
atoi = {'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,'K':19,'Ca':20,'Sc':21,'Ti':22,'Si':23,'V':24,'Cr':25}
atoc = {1: 40, 6: 100, 7: 150, 8: 200, 9:240}
KAYBEETEE = 0.000950048
BOHRPERA = 1.889725989

#
# -- begin Environment set up.
#
print("--------------------------\n")
print("         /\\______________")
print("      __/  \\   \\_________")
print("    _/  \\   \\            ")
print("___/\_TensorMol_0.0______")
print("   \\_/\\______  __________")
print("     \\/      \\/          ")
print("      \\______/\\__________\n")
print("--------------------------")
print("By using this software you accept the terms of the GNU public license in ")
print("COPYING, and agree to attribute the use of this software in publications as: \n")
print("K.Yao, J. Herr, J. Parkhill. TensorMol0.0 (2016)")
print("Depending on Usage, please also acknowledge, TensorFlow, PySCF, or your training sets.")
print("--------------------------")
print("Searching for Installed Optional Packages...")
try:
	from pyscf import scf
	from pyscf import gto
	from pyscf import dft
	from pyscf import mp
	HAS_PYSCF = True
	print("Pyscf has been found")
except Exception as Ex:
	print("Pyscf is not installed -- no ab-initio sampling",Ex)
	pass

try:
	import MolEmb
	HAS_EMB = True
	print("MolEmb has been found")
except:
	print("MolEmb is not installed. Please cd C_API; sudo python setup.py install")
	pass

try:
	import tensorflow as tf 
	HAS_TF = True
	print("Tensorflow has been found")
except:
	print("Tensorflow not Installed, very limited functionality")
	pass
print("TensorMol ready...")

SENSORYBASIS='''
C    S
	  1.0					1.0000000
C    S
	  0.5					1.0000000
C    S
	  0.1					1.0000000
C    S
	  0.05					1.0000000
C    P
	  1.0					1.0000000
C    P
	  0.1					1.0000000
C    P
	  0.05					1.0000000
C    P
	  0.025					1.0000000
C    D
	  0.25					1.0000000
C    D
	  0.125					1.0000000
C    D
	  0.0625				1.0000000
C    F
	  0.1					1.0000000
C    F
	  0.05					1.0000000
C    F
	  0.025					1.0000000
C    H
	  0.025					1.0000000
C    I
	  0.025					1.0000000
		'''
POTENTIAL_BASIS='''
C    S
	  10.0					1.0000000
C    S
	  1.0					1.0000000
C    S
	  0.1					1.0000000
C    S
	  0.01					1.0000000
C    P
	  100.0					1.0000000
C    P
	  10.0					1.0000000
C    P
	  1.0					1.0000000
C    D
	  100.0					1.0000000
C    D
	  10.0					1.0000000
C    D
	  1.0					1.0000000
C    D
	  0.1					1.0000000
		'''
TOTAL_SENSORY_BASIS=None
if (HAS_PYSCF):
	TOTAL_SENSORY_BASIS={'C': gto.basis.parse(SENSORYBASIS),'H@0': gto.basis.parse('''
H    S
	  1.0					1.0
		'''),'H@1': gto.basis.parse('''
H    S
	  0.2331359              1.0        
		'''),'H@6': gto.basis.parse('''
H    S
	  0.2883093              1.0
		'''),'H@7': gto.basis.parse('''
H    S
	  0.370571               1.0
		'''),'H@8': gto.basis.parse('''
H    S
	  0.4933630              1.0                   
		'''),'H@9': gto.basis.parse('''
H    S
	  0.4885885              1.0                  
		''')}

print("--------------------------")
#
# -- end Environment set up.
#

def scitodeci(sci):
	tmp=re.search(r'(\d+\.?\d+)\*\^(-?\d+)',sci)
	return float(tmp.group(1))*pow(10,float(tmp.group(2)))

def AtomicNumber(Symb):
	try:
		return atoi[Symb]
	except Exception as Ex:
		raise Exception("Unknown Atom")
	return 0

def AtomicSymbol(number):
	try:
		return atoi.keys()[atoi.values().index(number)]
	except Exception as Ex:
		raise Exception("Unknown Atom")
	return 0

def RotationMatrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def MakeUniform(point,disp,num):
	''' Uniform Grids of dim numxnumxnum around a point'''
	grids = np.mgrid[-disp:disp:num*1j, -disp:disp:num*1j, -disp:disp:num*1j]
	grids = grids.transpose()
	grids = grids.reshape((grids.shape[0]*grids.shape[1]*grids.shape[2], grids.shape[3]))
	return point+grids

def SignStep(S):
	if (S<0.5):
		return -1.0
	else:
		return 1.0

def MatrixPower(A,p):
	''' Raise a Hermitian Matrix to a possibly fractional power. '''
	#w,v=np.linalg.eig(A)
	# Use SVD
	u,s,v = np.linalg.svd(A)
	for i in range(len(s)):
		if (abs(s[i]) < np.power(10.0,-8)):
			s[i] == np.power(10.0,-8)
	#print("Matrixpower?",np.dot(np.dot(v,np.diag(w)),v.T), A)
	#return np.dot(np.dot(v,np.diag(np.power(w,p))),v.T)
	return np.dot(u,np.dot(np.diag(np.power(s,p)),v))

# Choose random samples near point...
def PointsNear(point,NPts,Dist):
	disps=Dist*0.2*np.abs(np.log(np.random.rand(NPts,3)))
	signs=signstep(np.random.random((NPts,3)))
	return (disps*signs)+point

def SamplingFunc_v2(S, maxdisp):    ## with sampling function f(x)=M/(x+1)^2+N; f(0)=maxdisp,f(maxdisp)=0; when maxdisp =5.0, 38 % lie in (0, 0.1)
	M = -((-1 - 2*maxdisp - maxdisp*maxdisp)/(2 + maxdisp))
	N = ((-1 - 2*maxdisp - maxdisp*maxdisp)/(2 + maxdisp)) + maxdisp
	return M/(S+1.0)**2 + N


def LtoS(l):
	s=""
	for i in l:
		s+=str(i)+" "
	return s	
	
		

def ErfSoftCut(dist, width, x):
	return (1-scipy.special.erf(1.0/width*(x-dist)))/2.0	

def CosSoftCut(dist, x):
	if x > dist:
		return 0
	else:
		return 0.5*(math.cos(math.pi*x/dist)+1.0)

signstep = np.vectorize(SignStep)
samplingfunc_v2 = np.vectorize(SamplingFunc_v2)
