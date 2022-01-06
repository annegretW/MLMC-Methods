import mc
import matplotlib.pyplot as plt
import numpy as np

#example functions k
def func_k1(x):
    return (-0.5*x+1)**(-2)/100

def func_k2(x):
    return (-x+1)**(-2)

def func_k3(x):
    return 1

#example functions p
def func_p1(x):
    return (-0.5*x+1)**3

def func_p2(x):
    return (-x+1)**3

def func_p3(x):
    return 1-x

#calculate eigenvalues for different parameters lambda
def eigenvalues_test(m_KL):
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
    
def discretization_test(func_k,func_p):
   # number of samples
    N = 100
    # truncation of KL-expansion
    m_KL = 100
    # Mesh size
    m = 8

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
    
def standard_montecarlo_test(N,m_KL,m,func_k,func_p):
    p_0 = func_p(0)
    p_1 = func_p(1)
    
    s = np.zeros(1000)
    for i in range(1000):
        s[i] = (2*i+1)/(2*1000)
        
    t = np.zeros(m)
    for i in range(m):
        t[i] = (2*i+1)/(2*m)

    r = np.zeros(1000)
    for i in range(1000):
        r[i] = func_p((2*i+1)/(2*1000))
    
    q = np.zeros(m)
    for i in range(m):
        q[i] = func_p((2*i+1)/(2*m))
        
    l = np.zeros(m)
    for i in range(m):
        l[i] = func_k((2*i+1)/(2*m))

    sol = mc.standard_mc(m_KL,m,N,func_k,p_0,p_1)  
    print(f"Der maximale Fehler beträgt {np.max(q-sol)}.")
    
    print(mc.k_eff(l,sol,0))
    print(mc.k_eff(l,q,0))

    plt.plot(t,sol)
    plt.plot(s,r)
    plt.legend(['mc','p'])
    plt.show()


def standard_mc_keff_test(N,m_KL,m,func_k,func_p):
    p_0 = func_p(0)
    p_1 = func_p(1)
    
    r = np.zeros(m)
    for i in range(m):
        r[i] = func_p((2*i+1)/(2*m))
        
    l = np.zeros(m)
    for i in range(m):
        l[i] = func_k((2*i+1)/(2*m))
    
    estimation = mc.standard_mc_new(m_KL,m,N,func_k,p_0,p_1,mc.k_eff)
    solution = mc.k_eff(l,r,p_1)
    print(estimation)
    print(solution)
    
    return abs(estimation-solution)
    
'''
1) Eigenwerte (Fig. 1 links)
'''
#eigenvalues_test(1000)


'''
2) Diskretisierung ist korrekt
'''
#discretization_test(func_k3,func_p3)


'''
3) Standard Montecarlo
'''
#standard_montecarlo_test(N=3,m_KL=800,m=8,func_k=func_k3,func_p=func_p3)


'''
4) Calculate k_eff with Standard Montecarlo
'''

#standard_mc_keff_test(N=100,m_KL=800,m=64,func_k=func_k3,func_p=func_p3)



#mc.mlmc(level=2,m_KL=80,m_0=16,N=[100,10],func_k=func_k,p_0=1,p_1=0.125)
m = 100
r = np.zeros(m)
for i in range(m):
    r[i] = func_p3((2*i+1)/(2*m))
        
l = np.zeros(m)
for i in range(m):
    l[i] = func_k3((2*i+1)/(2*m))
    
res = mc.k_eff(l,r,0)
print(f"Correct result: {res} \n")

#standard_mc = mc.standard_mc_new(800,16,100,func_k3,1,0,mc.k_eff)
#print(f"Standard MC \n Ergebnis: {standard_mc[0]}, Fehler: {abs(res-standard_mc[0])}, Kosten: {standard_mc[1]} \n")
#print(np.var(standard_mc[2]))
#print(abs(standard_mc[2]))

#multilevel_mc = mc.mlmc_new(100,16,func_k3,mc.k_eff,1,0,1e-3)
#print(f"Multilevel MC \n Ergebnis: {multilevel_mc}, Fehler: {abs(res-multilevel_mc)}")


multilevel_mc = mc.mlmc(100,4,[10,10,10],func_k3,mc.k_eff,1,0)
print(f"Multilevel MC \n Ergebnis: {multilevel_mc[0]}, Fehler: {abs(res-multilevel_mc[0])}, Kosten: {multilevel_mc[1]}")

n = len(multilevel_mc[2])
v = np.zeros(n)
x = np.zeros(n)

for i in range(len(multilevel_mc[2])):
    x[i] = i
    v[i] = mc.calc_variance(multilevel_mc[2][i])

w = np.zeros(n-1)
for i in range(n-1):
    w[i] = mc.calc_variance(np.array(multilevel_mc[2][i+1])-np.array(multilevel_mc[2][i]))
    
plt.plot(x,v)
plt.plot(x[1:],w)
plt.yscale('log', base=2)
plt.show()
