from __future__ import division
import numpy as np
from math import factorial, exp, sqrt
import data

y = data.getData('y')

def reducedOverlap(na, la, ma, nb, lb, alpha, beta):
	#reduced overlap integrals

	ss = 0.0

	#get the z coefficients
	z = data.getData('z')

	p = (alpha + beta) * 0.5
	pt= (alpha - beta) * 0.5
	m = abs(ma)

	#reverse quantum numbers if necessary
	if (lb < la) or ((lb == la) and (nb < na)):
		na,nb = nb, na
		la,lb = lb, la
		pt = -pt


	k = (na+nb-(la+lb)) % 2

	#get A and B integrals
	a = getAuxiliary_a(p, na+nb)
	b = getAuxiliary_b(pt, na+nb)

	#both s orbitals
	if (la == 0) and (lb == 0):
		z_index = int((90 - 17*na + pow(na, 2) - 2*nb)/2)

		for i in range(0, na+nb+1):
			n = na+nb-i
			ss += z[i,z_index-1]*a[i]*b[n]/2.0

		return ss

	else:
		y_index = int((5 - ma)*(24-10*ma+ma*ma)*(83-30*ma+3*ma*ma)/120 + \
		          (30-9*la+la*la-2*na)*(28-9*la+la*la-2*na)/8 + \
		          (30-9*lb+lb*lb-2*nb)/2)

		for i in range(9):
			for j in range(0, (4 - (k+i % 2))+1 ):
				ss += y[i,j,y_index-1]*a[i]*b[2*j + ((k+i) % 2)]

		return ss * pow(factorial(ma+1)/8.0, 2) * sqrt((2*la+1)*factorial(la-ma)*(2*lb+1)*factorial(lb-ma) / \
		            (4.0*factorial(la+ma)*factorial(lb+ma)))

def getAuxiliary_a(p, k):
	#the A auxiliary functions

	a = np.zeros((10))
	a[0] = exp(-p)/p

	for i in range(k):
		a[i+1] = (a[i]*(i+1) + exp(-p))/p

	return a

def getAuxiliary_b(p, k):
	#the B auxiliary functions

	b = np.zeros((10))

	if abs(p) > 3.0: type = ['e', 0]
	else:
		if (abs(p) > 2.0):
			if (k <= 10): type = ['e', 0]
			else: type = ['s', 16]
		elif (abs(p) > 1.0):
			if (k <= 7): type = ['e', 0]
			else: type = ['s', 13]
		elif (abs(p) > 0.5):
			if (k <= 5): type = ['e', 0]
			else: type = ['s', 8]
		elif (abs(p) > 1e-6): 
			type = ['s', 7]
		else:
			type = ['f', 0]

	if type[0] == 'e':
		b[0] = (exp(p) - exp(-p))/p
		for i in range(k):
			b[i+1] = (b[i]*(i+1) + exp(p) * pow(-1, i+1) - exp(-p))/p
	elif type[0] == 's':
		for i in range(0, k+1):
			y = 0.0
			for j in range(type[1]):
				y += pow(-p, j) * (1.0 - pow(-1.0, j+i+1))/(factorial(j) * (i+j+1))
			b[i] = y
	elif type[0] == 'f':
		for i in range(0, k+1):
			b[i] = (1.0 - pow(-1.0, (i+1)))/(i+1)

	return b


