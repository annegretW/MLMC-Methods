import numpy as np
from sympy import symbols, Eq, nsolve, solve, linsolve, solveset, tan, Reals, Interval
import math
import random
import scipy.integrate as integrate
import scipy.special as special

def draw_sample(theta,b,func_k,m,m_KL,sigma=None):
    '''
    Draws samples using the Karhunen-Loève expansion
    
        Parameters:
                theta (np.array(m)): Array of eigenvalues
                b (np.array(m,m_KL)): Array of eigenvectors
                func_k: Expected values of k
                m (int): Number of cells
                m_KL (int): Truncation point of KL expansion
        Returns:
                Sample k (vector of size m)
                Vector of random variables sigma (vector of size m_KL)
    '''
    k = np.zeros(m)

    if sigma is None:
        sigma = np.zeros(m_KL)
        for i in range(m_KL):
            sigma[i] = random.gauss(0, 1)
    for j in range(m):
        sum = 0
        for i in range(m_KL):
            sum += math.sqrt(theta[i])*sigma[i]*b[i,int((j/m)*m_KL)]
            #sum += math.sqrt(theta[i])*sigma[i]*b[i,j]
        k[j] = math.exp(math.log(func_k((2*j+1)/(2*m))) + sum)

    return k,sigma


def convert_sample(k):
    '''
    Converts a sample from level m to a sample of level m/2
    '''
    n = int(len(k)/2)
    res = np.zeros(n)
    for i in range(n):
        res[i] = (k[2*i]+k[2*i+1])/(2)
        #res[i] = k[2*i]
    return res   
    
def solve_pde(k,m,p_0,p_1):
    '''
    Calculates solution of pde
    
        Parameters:
                k (np.array(m)): Discrete values of k
                m (int): Number of cells
                p_0 (float): Lower boundary condition (Dirichlet)
                p_1 (float): Upper boundary condition (Dirichlet)
        Returns:
                Solution p of the pde as a np.array(m)
    '''    
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

def eigenpairs(m_KL,m):
    # parameters
    correlation_length = 0.3
    variance = 1

    w = np.zeros(m_KL)
    theta = np.zeros(m_KL)
    b = np.zeros([m_KL,m])

    x = symbols('x',real=True)
    expr = Eq((2*correlation_length*x)/((correlation_length**2)*(x**2)-1)-tan(x),0)

    w[0] = nsolve(expr,2)
    w[1] = nsolve(expr,4.5)
    for j in range(2,m_KL):
        w[j] = nsolve(expr,np.pi*j)

    for j in range(m_KL):
        theta[j] = 2*correlation_length / ((correlation_length**2)*(w[j]**2)+1)            

        for i in range(m):
            y = i/m
            h = math.sin(w[j]*y) + correlation_length*w[j]*math.cos(w[j]*y)
            b[j,i] = h/np.abs(h)

    return theta,b

def covariance(d,correlation_length,variance):
    return (variance**2)*np.exp(-np.linalg.norm(d)/correlation_length)
    
def eigenpairs_discrete(N,m,correlation_length,variance):
    C = np.zeros([N,N])
    
    for i in range(N):
        for j in range(N):
            C[i,j] = covariance(i/N-j/N,correlation_length,variance)/N

    theta, b = np.linalg.eig(C)
    
    return theta,b

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

def k_eff(k,p,p_1):
    #return k[1]
    return -k[-1]*(p_1-p[-1])*(2*len(p))
    #n = len(p)
    #return (p[int((n/2))]+p[int(n/2)-1])/2

# calculates the standard Monte-Carlo estimator
def standard_mc(m_KL,m,N,func_k,p_0,p_1,Q):
    c = 0
    theta, b = eigenpairs_discrete(m_KL,m,0.3,1)
    sum = 0
    Q_list = np.zeros(N)
    for i in range(N):
        k = draw_sample(theta,b,func_k,m,m_KL)[0]
        q = Q(k,solve_pde(k,m,p_0,p_1),p_1)
        Q_list[i] = q
        sum += q
        c += m
        
    return sum/N, c, Q_list

# calculate the multilevel Monte-Carlo estimator
def mlmc(m_KL,m_0,N,func_k,Q,p_0,p_1):
    c = 0 #costs
    level = len(N) #number of levels
     
    estimator = 0 
    Y_list = [] 
    Q_list = [[]]
    k_list = [[]]
    
    #Calculate Q on finest mesh
    m = (2**(level-1))*m_0 #mesh size
    theta, b = eigenpairs_discrete(m_KL,m,0.3,1)
    #theta, b = eigenpairs(m_KL,m)
    for i in range(N[level-1]):
        sample = draw_sample(theta,b,func_k,m,m_KL)[0]
        k_list[0].append(sample)
        Q_list[0].append(Q(sample,solve_pde(sample,m,p_0,p_1),p_1))
        c += m
              
    #go through all meshs
    for i in range(level-1,0,-1):
        m = (2**(i-1))*m_0
        
        Q_list.insert(0,[])
        Y_list.insert(0,[])
        k_list.insert(0,[])
        sum = 0
        for j in range(N[i]):
            sample = convert_sample(k_list[1][j])
            k_list[0].append(sample)
            Q_list[0].append(Q(sample,solve_pde(sample,m,p_0,p_1),p_1))
            Y_list[0].append(Q_list[1][j]-Q_list[0][j])
            c += m
            sum += (Q_list[1][j]-Q_list[0][j])
          
        estimator += (sum/N[i])
        
        for j in range(N[i],N[i-1]):
            sample = draw_sample(theta,b,func_k,m,m_KL)[0]
            k_list[0].append(sample)
            Q_list[0].append(Q(sample,solve_pde(sample,m,p_0,p_1),p_1))
            c += m
                    
    #Calculate Q on coarsest mesh
    sum = 0
    for i in range(N[0]):
        sum += Q_list[0][i]
    estimator += (sum/N[0])
    
    return estimator, c, Q_list, Y_list, k_list

# calculate the multilevel Monte-Carlo estimator
def mlmc_changed(m_KL,m_0,N,func_k,Q,p_0,p_1):
    c = 0 #costs
    level = len(N) #number of levels
     
    estimator = 0 
    Y_list = [] 
    Q_list = [[]]
    k_1 = []
    
    #Calculate Q on finest mesh
    m = (2**(level-1))*m_0 #mesh size
    theta, b = eigenpairs_discrete(m_KL,m,0.3,1)
    for i in range(N[level-1]):
        sample = draw_sample(theta,b,func_k,m,m_KL)[0]
        k_1.append(sample)
        Q_list[0].append(Q(sample,solve_pde(sample,m,p_0,p_1),p_1))
        c += m
    
    #go through all meshs
    for i in range(level-1,0,-1):
        m = (2**(i-1))*m_0
        
        Q_list.insert(0,[])
        Y_list.insert(0,[])     
        sum = 0
        for j in range(N[i]):
            sample = convert_sample(k_1[j])
            Q_new = Q(sample,solve_pde(sample,m,p_0,p_1),p_1)
            Y_list[0].append(Q_list[1][j]-Q_new)
            c += m
            sum += (Q_list[1][j]-Q_new)
          
        estimator += (sum/N[i])
        
        k_1 = []
        for j in range(N[i-1]):
            sample = draw_sample(theta,b,func_k,m,m_KL)[0]
            k_1.append(sample)
            Q_list[0].append(Q(sample,solve_pde(sample,m,p_0,p_1),p_1))
            c += m
        
    #Calculate Q on coarsest mesh
    sum = 0
    for i in range(N[0]):
        sum += Q_list[0][i]
    estimator += (sum/N[0])
        
    return estimator, c, Q_list, Y_list

# calculate the multilevel Monte-Carlo estimator
def mlmc_test(m_KL,m_0,N,func_k,Q,p_0,p_1):
    c = 0 
    level = len(N)
     
    estimator = 0
    Y_list = []
    Q_list = [[]]
    #k_list = [[]]
    sample_list = []
    
    #Calculate Q on finest mesh
    #print(f"Calculate Q_l for l = {level-1}")
    m = (2**(level-1))*m_0
    theta, b = eigenpairs_discrete(m_KL,m,0.8,1)
    for i in range(N[level-1]):
        sample = draw_sample(theta,b,func_k,m,m_KL)
        #k_list[0].append(sample)
        sample_list.append(sample[1])
        Q_list[0].append(Q(sample[0],solve_pde(sample[0],m,p_0,p_1),p_1))
        c += m
              
    #go through all meshs
    for i in range(level-1,0,-1):
        #print(f"\nCalculate Y_l for l = {i}")
        m = (2**(i-1))*m_0
        
        Q_list.insert(0,[])
        Y_list.insert(0,[])
        #k_list.insert(0,[])
        sum = 0
        for j in range(N[i]):
            #sample = convert_sample(k_list[1][j])
            sample = draw_sample(theta,b,func_k,m,m_KL,sample_list[j])
            #k_list[0].append(sample)
            Q_list[0].append(Q(sample[0],solve_pde(sample[0],m,p_0,p_1),p_1))
            Y_list[0].append(Q_list[1][j]-Q_list[0][j])
            c += m
            sum += (Q_list[1][j]-Q_list[0][j])
        estimator += (sum/N[i])
        
        #print(f"\nCalculate Q_l for l = {i-1}")
        for j in range(N[i],N[i-1]):
            sample = draw_sample(theta,b,func_k,m,m_KL)
            #k_list[0].append(sample)
            sample_list.append(sample[1])
            Q_list[0].append(Q(sample[0],solve_pde(sample[0],m,p_0,p_1),p_1))
            c += m
                    
    #Calculate Q on coarsest mesh
    sum = 0
    #print("\nCalculate Y_l for l = 0")
    for i in range(N[0]):
        sum += Q_list[0][i]
    estimator += (sum/N[0])
    
    return estimator, c, Q_list, Y_list

# calculate the multilevel Monte-Carlo estimator
def mlmc_new(m_KL,M,func_k,Q,p_0,p_1,eps):
    costs = 0 
     
    estimator = 0
    est_list = []
    Q_list = [[]]
    k_list = [[]]
    
    '''
    Step 1
    '''
    level = 0 #start with L=0
    
    '''
    Step 2
    '''
    #initial number of samples on Level L
    N = [100]
    
    #Calculate Q on finest mesh 
    m = M
    theta, b = eigenpairs_discrete(m_KL,m,0.3,1,max(m,m_KL))
    for i in range(N[-1]):
        sample = draw_sample(theta,b,func_k,m,m_KL)
        k_list[0].append(sample)
        Q_list[0].append(Q(sample,solve_pde(sample,m,p_0,p_1),p_1))
        costs += m
    
    #calculate mean
    mean = 0
    for i in range(N[-1]):
        mean += Q_list[0][i]
    mean = mean/N[-1]
    
    #calculate variance
    var = 0
    for i in range(N[-1]):
        var += (Q_list[0][i]-mean)**2
    var = var/N[-1]
    
    '''
    Step 3
    '''
    c = 2/(eps**2)*math.sqrt(m*var)
    N_L = c*math.sqrt(var/m)
    N[-1] = int(N_L)
    
    print(N[-1])
    for i in range(len(Q_list[0]),N[-1]):
        sample = draw_sample(theta,b,func_k,m,m_KL)
        k_list[0].append(sample)
        Q_list[0].append(Q(sample,solve_pde(sample,m,p_0,p_1),p_1))
        costs += m
    
    estimator = sum(Q_list[0])/N[-1]
    
    '''   
    m = int(m/2)
    Q_list.insert(0,[])
    k_list.insert(0,[])
    for i in range(N[-1]):
        sample = convert_sample(k_list[1][i])
        k_list[0].append(sample)
        Q_list[0].append(Q(sample,solve_pde(sample,m,p_0,p_1),p_1))
        costs += m
    
    #calculate mean
    mean = 0
    for i in range(N[-1]):
        print((Q_list[1][i]-Q_list[0][i]))
        mean += (Q_list[1][i]-Q_list[0][i])
    mean = mean/N[-1]
    
    #calculate variance
    var = 0
    for i in range(N[-1]):
        var += ((Q_list[1][i]-Q_list[0][i])-mean)**2
    var = var/N[-1]
    
    N_L = math.sqrt(var/m)
    
    print(mean)
    print(var)
    print(N_L)
    '''
    
    '''
    #Calculate Q on finest mesh
    m = (2**(level))*m_0
    theta, b = eigenpairs_discrete(m_KL,m,0.3,1,max(m,m_KL))
    for i in range(N[-1]):
        sample = draw_sample(theta,b,func_k,m,m_KL)
        k.append(sample)
        Q_list[0].append(Q(sample,solve_pde(sample,m,p_0,p_1),p_1))
        costs += m
      
    #go through all meshs
    for i in range(level-1,0,-1):
        #print(f"Level: {i+1}, Q = {Q_coarse_level}")
        m = (2**i)*m_0
        
        Q_list.insert(0,[])
        sum = 0
        for j in range(N[i]):
            Q_list[0].append(Q(k[j],solve_pde(k[j],m,p_0,p_1),p_1))
            costs += m
            sum += (Q_list[0][j]-Q_list[1][j])
        estimator += (sum/N[i])
        
        for j in range(N[i],N[i-1]):
            sample = draw_sample(theta,b,func_k,m,m_KL)
            k.append(sample)
            Q_list[0].append(Q(sample,solve_pde(sample,m,p_0,p_1),p_1))
            c += m
        
    #Calculate Q on coarsest mesh
    sum = 0
    for i in range(N[1]):
        sum += Q_list[0][i]
    for i in range(N[1],N[0]):
        sample = draw_sample(theta,b,func_k,m_0,m_KL)
        sum += Q(sample,solve_pde(sample,m_0,p_0,p_1),p_1)
        costs += m_0
    estimator += (sum/N[0])
    '''
    print(f"Kosten: {costs}")
    return estimator

