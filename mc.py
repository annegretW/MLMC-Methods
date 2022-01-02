import numpy as np
from sympy import symbols, Eq, nsolve, solve, linsolve, solveset, tan, Reals, Interval
import math
import random
import roots
import scipy.integrate as integrate
import scipy.special as special

def draw_sample(theta,b,func_k,m,m_KL):
    k = np.zeros(m)

    for j in range(m):
        sum = 0
        for i in range(m_KL):
            sigma = random.gauss(0, 1)
            sum += math.sqrt(theta[i])*sigma*b[i,j]
        k[j] = math.exp(math.log(func_k((2*j+1)/(2*m))) + sum)
        
    return k


def solve_pde(k,m,p_0,p_1):
    # initialize f
    f = np.zeros(m)
    f[0] = 2*k[0]*p_0
    f[m-1] = 2*k[m-1]*p_1
    
    # Create matrix A
    A = np.zeros((m,m))
    A[0,0] = 2*k[0] + 2/(1/k[1]+1/k[0])
    A[0,1] = -2/(1/k[0]+1/k[1])

    for i in range(1,m-1):
        A[i,i-1] = -2/(1/k[i-1]+1/k[i])
        A[i,i] = 2/(1/k[i-1]+1/k[i]) + 2/(1/k[i+1]+1/k[i])
        A[i,i+1] = -2/(1/k[i+1]+1/k[i])
        
    A[m-1,m-2] = -2/(1/k[m-2]+1/k[m-1])
    A[m-1,m-1] = 2*k[m-1] + 2/(1/k[m-1]+1/k[m-2])

    # Solve linear system
    p = np.linalg.solve(A, f)
    return p


def eigenpairs(m_KL,m,correlation_length=0.3,variance=1):
    w = np.zeros(m_KL)
    theta = np.zeros(m_KL)
    b = np.zeros([m_KL,m])

    f = lambda x:(2*correlation_length*x)/((correlation_length**2)*(x**2)-1)-tan(x)
    w = roots.roots(f,0,1000)[:m_KL]
                  
    for j in range(m_KL):
        theta[j] = 2*correlation_length / ((correlation_length**2)*(w[j]**2)+1)            
        
        for i in range(m):
            y = i/m
            h = math.sin(w[j]*y) + correlation_length*w[j]*math.cos(w[j]*y)
            b[j,i] = h/np.abs(h)

    return theta,b

def covariance(d,correlation_length,variance):
    return (variance**2)*np.exp(-np.linalg.norm(d)/correlation_length)
    
def eigenpairs_discrete(m_KL,m,correlation_length,variance,N):
    w = np.zeros(m_KL)
    theta = np.zeros(m_KL)
    b = np.zeros([m_KL,m])

    C = np.zeros([N,N])
    
    for i in range(N):
        for j in range(N):
            C[i,j] = covariance(i/N-j/N,correlation_length,variance)/N
            
    theta, b = np.linalg.eig(C)

    return theta[:m_KL],b[:m_KL]

def eigenpairs_numerical(m_KL,m,correlation_length,variance):
    w = np.zeros(m_KL)
    theta = np.zeros(m_KL)
    b = np.zeros([m_KL,m])

    x = symbols('x',real=True)
    expr = Eq((2*correlation_length*x)/((correlation_length**2)*(x**2)-1)-tan(x),0)
        
    w[0] = 0
    for j in range(1,m_KL):
        k = 3
        while k<5:
            try:
                r = nsolve(expr,w[j-1]+k)
                if np.abs(r-w[j-1])<10e-5:
                    k = k+0.1
                else:
                    w[j] = r
                    break
            except:
                k = k+1
            
    for j in range(m_KL):
        theta[j] = 2*correlation_length / ((correlation_length**2)*(w[j]**2)+1)
        
        for i in range(m):
            y = i/m
            h = math.sin(w[j]*y) + correlation_length*w[j]*math.cos(w[j]*y)
            if h!=0:
                b[j,i] = h/np.abs(h)
            else:
                b[j,i] = 0

    return theta,b

# calculate the standard Monte-Carlo estimator
def standard_mc(m_KL,m,N,func_k,p_0,p_1):
    theta, b = eigenpairs_discrete(m_KL,m,0.3,1,max(m,m_KL))
    sum = 0
    for i in range(N):
        k = draw_sample(theta,b,func_k,m,m_KL)
        sum += solve_pde(k,m,p_0,p_1)
    return sum/N

def k_eff(k,p,p1):
    print(k)
    print((p[-1]-0)*(len(p)-1))
    return -k[-1]*(p[-1]-0)*(len(p)-1)

# calculate the standard Monte-Carlo estimator
def standard_mc_new(m_KL,m,N,func_k,p_0,p_1,Q):
    theta, b = eigenpairs_discrete(m_KL,m,0.3,1,max(m,m_KL))
    sum = 0
    for i in range(N):
        k = draw_sample(theta,b,func_k,m,m_KL)
        sum += Q(k,solve_pde(k,m,p_0,p_1),0)
    return sum/N

# calculate the multilevel Monte-Carlo estimator
def mlmc(level,m_KL,m_0,N,func_k,p_0,p_1):
    m = np.zeros(level,int)
    m[0] = m_0
    
    theta, b = eigenpairs(m_KL,m_0)    
    sum = 0  
    for i in range(N[0]):
        k = draw_sample(theta,b,func_k,m_0,m_KL)
        sum += solve_pde(k,m_0,p_0,p_1)
    estimator = sum/N[0]
    
    for i in range(1,level):
        m[i] = (2**i)*m_0
        theta, b = eigenpairs(m_KL,m[i])  
        sum = 0  
        for i in range(N[i]):
            k = draw_sample(theta,b,func_k,m[i],m_KL)
            sum += (solve_pde(k,m[i],p_0,p_1)-solve_pde(k,m[i-1],p_0,p_1))
        estimator += sum/N[i]
