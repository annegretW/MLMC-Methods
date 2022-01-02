import mc
import matplotlib.pyplot as plt
import numpy as np

#example function k
def func_k(x):
    return (-x+1)**(-2)

#example function p
def func_p(x):
    return (-x+1)**3

#calculate eigenvalues for different parameters lambda
def eigenvalues(m_KL):
    fig, ax = plt.subplots()
    t = np.linspace(1,m_KL,m_KL)
    
    theta,b = mc.eigenpairs_discrete(m_KL,10,1,1)
    ax.loglog(t, theta, label='$\lambda=1$')
    
    theta,b = mc.eigenpairs_discrete(m_KL,10,0.1,1)
    ax.loglog(t, theta, label='$\lambda=0.1$')
    
    theta,b = mc.eigenpairs_discrete(m_KL,10,0.01,1)
    ax.loglog(t, theta, label='$\lambda=0.01$')
    
    plt.xlabel('n')
    plt.ylabel(r'Eigenwerte $\theta_n$')
    plt.legend()
    plt.show()
    
def discretization():
   # number of samples
    N = 100
    # truncation of KL-expansion
    m_KL = 800
    # Mesh size
    m = 100

    # Dirichlet boundary conditions
    p_0 = 1
    p_1 = 0

    s = np.zeros(1000)
    for i in range(1000):
        s[i] = (2*i+1)/(2*1000)
        
    t = np.zeros(m)
    for i in range(m):
        t[i] = (2*i+1)/(2*m)
        
    r = np.zeros(1000)
    for i in range(1000):
        r[i] = func_p((2*i+1)/(2*1000))
        
    l = np.zeros(m)
    for i in range(m):
        l[i] = func_k((2*i+1)/(2*m))


    plt.plot(t,mc.solve_pde(l,m,p_0,p_1))
    plt.plot(s,r)
    plt.legend(['approximation','p'])
    plt.show() 

'''
1) Eigenwerte (Fig. 1 links)

eigenvalues(1000)
'''


'''
2) Diskretisierung ist korrekt

discretization()
'''

'''
3) Standard Montecarlo

# number of samples
N = 1000
# truncation of KL-expansion
m_KL = 800
# Mesh size
m = 16

# Dirichlet boundary conditions
p_0 = 1
p_1 = 0

s = np.zeros(1000)
for i in range(1000):
    s[i] = (2*i+1)/(2*1000)
    
t = np.zeros(m)
for i in range(m):
    t[i] = (2*i+1)/(2*m)

r = np.zeros(1000)
for i in range(1000):
    r[i] = func_p((2*i+1)/(2*1000))
    
l = np.zeros(m)
for i in range(m):
    l[i] = func_k((2*i+1)/(2*m))

plt.plot(t,mc.standard_mc(m_KL,m,N,func_k,p_0,p_1))
plt.plot(s,r)
plt.legend(['mc','p'])
plt.show()
'''

# number of samples
N = 100
# truncation of KL-expansion
m_KL = 800
# Mesh size
m = 100

# Dirichlet boundary conditions
p_0 = 1
p_1 = 0

r = np.zeros(m)
for i in range(m):
    r[i] = func_p((2*i+1)/(2*m))
    
l = np.zeros(m)
for i in range(m):
    l[i] = func_k((2*i+1)/(2*m))
    
print(mc.standard_mc_new(m_KL,m,N,func_k,p_0,p_1,mc.k_eff))
print(mc.k_eff(l,r,0))
