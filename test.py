import numpy as np
import matplotlib.pyplot as plt
import mc

# number of samples
N = 100
# truncation of KL-expansion
m_KL = 800
# Mesh size
m = 100

# Dirichlet boundary conditions
p_0 = 1
p_1 = 0
    
def func_k(x):
    return (-x+1)**(-2)
    
def func_p(x):
    return (-x+1)**3

t = np.zeros(m)
for i in range(m):
    t[i] = (2*i+1)/(2*m)
    
r = np.zeros(m)
for i in range(m):
    r[i] = func_p((2*i+1)/(2*m))
    
l = np.zeros(m)
for i in range(m):
    l[i] = func_k((2*i+1)/(2*m))


plt.plot(t,mc.solve_pde(l,m,p_0,p_1))
plt.plot(t,r)
plt.legend(['approximation','p'])
plt.show()


'''
estimator = mc.standard_mc(m_KL,8,N,func_k,p_0,p_1)
t = np.zeros(8)
for i in range(8):
    t[i] = (2*i+1)/(2*8)
plt.plot(t,estimator)
estimator = mc.standard_mc(m_KL,16,N,func_k,p_0,p_1)
t = np.zeros(16)
for i in range(16):
    t[i] = (2*i+1)/(2*16)
plt.plot(t,estimator)
estimator = mc.standard_mc(m_KL,32,N,func_k,p_0,p_1)
t = np.zeros(32)
for i in range(32):
    t[i] = (2*i+1)/(2*32)
plt.plot(t,estimator)
'''
'''
mlmc_estimator = mc.mlmc(2,m_KL,8,[100,100],func_k,p_0,p_1)

s = np.linspace(0,1,1000)
plt.plot(s,func_p(s))
plt.legend(['m=8','m=16','m=32','p'])
plt.show()'''