import mc
import math
import matplotlib.pyplot as plt
import numpy as np

#example for expected value of k
def func_k1(x):
    return (-0.5*x+1)**(-2)/100

def func_k2(x):
    return (-x+1)**(-2)

def func_k3(x):
    return 50

def func_k4(x):
    if(x<0.2):
        return 20
    elif(x<0.4):
        return 30
    elif(x<0.6):
        return 10
    elif(x<0.8):
        return 50
    else:
        return 80
    
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
 
def eigenvalues_sum(m_KL):
    fig, ax = plt.subplots()
    t = np.linspace(1,m_KL,m_KL)
    
    theta,b = mc.eigenpairs_discrete(m_KL,10,1,1)
    x = np.zeros(m_KL)
    x[0] = theta[0]
    for i in range(1,m_KL):
        x[i] = x[i-1]+theta[i]
    ax.loglog(t, x, label='$\lambda=1$')
    
    theta,b = mc.eigenpairs_discrete(m_KL,10,0.1,1)
    y = np.zeros(m_KL)
    y[0] = theta[0]
    for i in range(1,m_KL):
        y[i] = y[i-1]+theta[i]
    ax.loglog(t, y, label='$\lambda=0.1$')
    
    theta,b = mc.eigenpairs_discrete(m_KL,10,0.01,1)
    z = np.zeros(m_KL)
    z[0] = theta[0]
    for i in range(1,m_KL):
        z[i] = z[i-1]+theta[i]
    ax.loglog(t, z, label='$\lambda=0.01$')
    
    plt.xlabel('n')
    plt.ylabel(r'Summe der ersten n Eigenwerte')
    plt.legend()
    plt.show()
    
'''    
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
'''

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
    #print(f"Der maximale Fehler betraegt {np.max(q-sol)}.")
    
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
    
    estimation = mc.standard_mc(m_KL,m,N,func_k,p_0,p_1,mc.k_eff)[0]
    solution = mc.k_eff(l,r,p_1)
    #print(estimation)
    #print(solution)
    print(abs(estimation-solution))
    return abs(estimation-solution)
 
def error_KL(m_KL,m,func_k,correlation_length,variance):
    theta, b = mc.eigenpairs_discrete(50,m,correlation_length,variance)
    result = mc.k_eff(mc.draw_sample(theta,b,func_k,m,50),sol)
    
    truncated_result = []
    for i in range(len(m_KL)):
        h = mc.k_eff(draw_sample(theta,b,func_k,m,m_KL[i]))
        truncated_result.append((result-k)/result)
    
def constant_variance(N,m_KL,m,func_k):
    standard_mc = []
    for i in range(len(m)):
        standard_mc.append(mc.standard_mc(m_KL,m[i],N,func_k3,1,0,mc.k_eff)[2])
    
    v_mean = np.zeros(len(m))
    v_var = np.zeros(len(m))
    x = np.zeros(len(m))
    for i in range(len(m)):
        x[i] = i
        v_mean[i] = math.log2(abs(np.mean(standard_mc[i])))
        v_var[i] = math.log2(abs(np.var(standard_mc[i])))
        print(abs(np.var(standard_mc[i])))
    '''    
    plt.plot(x,v_mean)
    plt.xlabel('Level l')
    plt.ylabel(r'$\log_2$-Erwartungswert')
    plt.show()
    
    plt.plot(x,v_var)
    plt.xlabel('Level l')
    plt.ylabel(r'$\log_2$-Varianz')
    plt.show()
'''
    
'''
1) Eigenwerte (Fig. 1 links)
'''
#eigenvalues_test(1000)
#eigenvalues_sum(1000)


'''
2) Diskretisierung ist korrekt
'''
#discretization_test(func_k3,func_p3)


'''
3) Standard Montecarlo
'''
#standard_montecarlo_test(N=3,m_KL=800,m=128,func_k=func_k3,func_p=func_p3)


'''
4) Calculate k_eff with Standard Montecarlo
'''
#standard_mc_keff_test(N=300,m_KL=800,m=64,func_k=func_k3,func_p=func_p3)


'''
5) Variance of Q_M is independent of M
'''
#constant_variance(N=100,m_KL=800,m=[16,32,64,128,256],func_k=func_k3)


print("\n ....................................................... \n")

multilevel_mc = mc.mlmc(800,16,[100,100,100,100,100],func_k3,mc.k_eff,1,0)
print(math.log2(abs(np.var(multilevel_mc[3][0]))))
print(math.log2(abs(np.var(multilevel_mc[3][1]))))
print(math.log2(abs(np.var(multilevel_mc[3][2]))))
print(math.log2(abs(np.var(multilevel_mc[3][3]))))

standard_mc = mc.standard_mc(800,16,100,func_k3,1,0,mc.k_eff)
print(math.log2(abs(np.var(standard_mc[2]))))

standard_mc = mc.standard_mc(800,32,100,func_k3,1,0,mc.k_eff)
print(math.log2(abs(np.var(standard_mc[2]))))

standard_mc = mc.standard_mc(800,64,100,func_k3,1,0,mc.k_eff)
print(math.log2(abs(np.var(standard_mc[2]))))

standard_mc = mc.standard_mc(800,128,100,func_k3,1,0,mc.k_eff)
print(math.log2(abs(np.var(standard_mc[2]))))

standard_mc = mc.standard_mc(800,64,100,func_k3,1,0,mc.k_eff)
print(math.log2(abs(np.var(standard_mc[2]))))

'''
#print(f"Multilevel MC \n Ergebnis: {multilevel_mc[0]}, Fehler: {abs(res-multilevel_mc[0])}, Kosten: {multilevel_mc[1]}")

n = len(multilevel_mc[2])

v_mean = np.zeros(n)
v_var = np.zeros(n)
x = np.zeros(n)
for i in range(n):
    x[i] = i
    v_mean[i] = math.log2(abs(np.mean(multilevel_mc[2][i])))
    v_var[i] = math.log2(abs(np.var(multilevel_mc[2][i])))
    print(v_var[i])

print('-----------------------------------------')

w_mean = np.zeros(n-1)
w_var = np.zeros(n-1)
for i in range(n-1):
    w_mean[i] = math.log2(abs(np.mean(multilevel_mc[3][i])))
    w_var[i] = math.log2(abs(np.var(multilevel_mc[3][i])))
    print(w_var[i])

plt.plot(x,v_mean)
plt.plot(x[1:],w_mean)
plt.legend([r'$Q_l$', '$Y_l$'])
plt.xlabel('Level l')
plt.ylabel(r'$\log_2$-Erwartungswert')
plt.show()

plt.plot(x,v_var)
plt.plot(x[1:],w_var)
plt.legend([r'$Q_l$', '$Y_l$'])
plt.xlabel('Level l')
plt.ylabel(r'$\log_2$-Varianz')
plt.show()
'''

#mc.mlmc_algo(800,16,func_k3,mc.k_eff,1,0,1e-2,1.75)
#error_KL([10,50,100,150],100,func_k3,0.3,1)
