import numpy as np
from sympy import symbols, Eq, nsolve, solve, linsolve, solveset, tan, Reals, Interval
import math
import random
import scipy.integrate as integrate
import scipy.special as special

def draw_sample(theta,b,func_k,m,m_KL,sigma=None):
    '''
    Draws samples using the Karhunen-Loeve expansion
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
    n = m
    k = np.zeros(n)

    if sigma is None:
        sigma = np.zeros(m_KL)
        for i in range(m_KL):
            sigma[i] = random.gauss(0, 1)
    for j in range(n):
        sum = 0
        for i in range(m_KL):
            sum += math.sqrt(theta[i])*sigma[i]*b[i,int((j/n)*m_KL)]
            #sum += math.sqrt(theta[i])*sigma[i]*b[i,j]
        k[j] = math.exp(math.log(func_k((2*j+1)/(2*n))) + sum)

    #k = convert_sample(k,m)
    return k,sigma


def convert_sample(k):
    '''
    Converts a sample from level m to a sample of level m/2
    '''
    n = int(len(k)/2)
    res = np.zeros(n)
    for i in range(n):
        res[i] = (k[2*i]+k[2*i+1])/(2)
    return res

def convert_sample(k,m):
    '''
    Converts a sample from level len(k) to a sample of level m
    '''
    n = len(k)
    l = n/m 
    res = np.zeros(m)
    for i in range(m):
        res[i] = (k[int(i*l+l/2-1)]+k[int(i*l+l/2)])/(2)
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

def covariance(d,correlation_length,variance):
    return (variance**2)*np.exp(-np.linalg.norm(d)/correlation_length)
    
def eigenpairs_discrete(N,correlation_length,variance):
    C = np.zeros([N,N])
    
    for i in range(N):
        for j in range(N):
            C[i,j] = covariance(i/N-j/N,correlation_length,variance)/N

    theta, b = np.linalg.eig(C)
    
    return theta,b

def k_eff(k,p,p_1):
    return -k[-1]*(p_1-p[-1])*(2*len(p))

# calculates the standard Monte-Carlo estimator
def standard_mc(m_KL,m,N,func_k,p_0,p_1,Q):
    c = 0
    theta, b = eigenpairs_discrete(max(m,m_KL),0.3,1)
    sum = 0
    Q_list = np.zeros(N)
    for i in range(N):
        k = draw_sample(theta,b,func_k,m,m_KL)[0]
        q = Q(k,solve_pde(k,m,p_0,p_1),p_1)
        Q_list[i] = q
        sum += q
        c += m
        
    print(sum/N)
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
    theta, b = eigenpairs_discrete(max(m_KL,m),0.3,1)
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
            sample = convert_sample(k_list[1][j],m)
            k_list[0].append(sample)
            Q_list[0].append(Q(sample,solve_pde(sample,m,p_0,p_1),p_1))
            #Q_new = Q(sample,solve_pde(sample,m,p_0,p_1),p_1)
            Y_list[0].append(Q_list[1][j]-Q_list[0][j])
            #Y_list[0].append(Q_list[1][j]-Q_new)
            c += m
            sum += (Q_list[1][j]-Q_list[0][j])
            #sum += (Q_list[1][j]-Q_new)
          
        estimator += (sum/N[i])
        
        #for j in range(N[i],N[i-1]):
        for j in range(N[i-1]):
            sample = draw_sample(theta,b,func_k,m,m_KL)[0]
            k_list[0].append(sample)
            Q_list[0].append(Q(sample,solve_pde(sample,m,p_0,p_1),p_1))
            c += m
                    
    #Calculate Q on coarsest mesh
    sum = 0
    for i in range(N[0]):
        sum += Q_list[0][i]
    estimator += (sum/N[0])
    
    print(estimator)
    return estimator, c, Q_list, Y_list, k_list

# calculate the multilevel Monte-Carlo estimator
def mlmc_changed(m_KL,m_0,N,func_k,Q,p_0,p_1):
    c = 0 #costs
    level = len(N) #number of levels
     
    estimator = 0 
    Y_list = []
    k_list = [] 
    Q_list = [[]]
    
    #Calculate Q on finest mesh
    m = (2**(level-1))*m_0
    theta, b = eigenpairs_discrete(max(m_KL,m),0.3,1)
    for i in range(N[level-1]):
        sample = draw_sample(theta,b,func_k,2*m,m_KL)[0]
        k_list.append(sample)
        sample_conv = convert_sample(sample,m)
        Q_list[0].append(Q(sample_conv,solve_pde(sample_conv,m,p_0,p_1),p_1))
        c += m
              
    #go through all meshs
    for i in range(level-1,0,-1):
        m = (2**(i-1))*m_0
        
        Q_list.insert(0,[])
        Y_list.insert(0,[])
        #summe = 0
        for j in range(N[i]):
            sample = convert_sample(k_list[j],m)
            Q_list[0].append(Q(sample,solve_pde(sample,m,p_0,p_1),p_1))            
            #Q_new = Q(sample,solve_pde(sample,m,p_0,p_1),p_1)
            Y_list[0].append(Q_list[1][j]-Q_list[0][j])
            #Y_list[0].append(Q_list[1][j]-Q_new)
            c += m
            #summe += (Q_list[1][j]-Q_list[0][j])
            #sum += (Q_list[1][j]-Q_new)
          
        #estimator += (sum/N[i])
        #print(sum/N[i])
        
        for j in range(N[i],N[i-1]):
            #for j in range(N[i-1]):
            sample = draw_sample(theta,b,func_k,m,m_KL)[0]
            #sample = convert_sample(sample,m)
            k_list.append(sample)
            Q_list[0].append(Q(sample,solve_pde(sample,m,p_0,p_1),p_1))
            c += m
                    
    #Calculate Q on coarsest mesh
    #summe = 0
    #for i in range(N[0]):
    #    summe += Q_list[0][i]
        
    Y_list.insert(0,Q_list[0])

    estimator = 0
    for j in range(len(Y_list)):
        estimator += sum(Y_list[j])/len(Y_list[j])
        print(estimator)
        
    return estimator, c, Q_list, Y_list, k_list

# calculate the multilevel Monte-Carlo estimator
def mlmc_algo(m_KL,M,func_k,Q,p_0,p_1,eps,alpha):
    costs = 0
    initial_number = 20
     
    Y_list = []
    Q_list = []
    k_list = []
    N = []
    
    '''
    Step 1
    Start with L=0.
    '''
    L = 0
    
    while(1):
        '''
        Step 2
        '''
        #initial number of samples on Level L
        level = L+1
        N.append(initial_number)
   
        Q_list.append([])
        Y_list.append([])
        k_list.append([])
        
        #Calculate Q on level L
        m = (2**(L))*M 
        theta, b = eigenpairs_discrete(max(m_KL,m),0.3,1)
        for i in range(N[level-1]):
            sample = draw_sample(theta,b,func_k,2*m,m_KL)[0]
            sample = convert_sample(sample,m)
            k_list[L].append(sample)
            Q_list[L].append(Q(sample,solve_pde(sample,m,p_0,p_1),p_1))
            costs += m
         
        #calculate Y_L for each sample
        if L==0:
            for i in range(N[L]):
                Y_list[L].append(Q_list[L][i])
        else:
            for j in range(N[L]):
                sample = convert_sample(k_list[L][j],int(m/2))
                k_list[L-1].append(k_list[L][j])
                Q_list[L-1].append(Q(sample,solve_pde(sample,int(m/2),p_0,p_1),p_1))
                Y_list[L].append(Q_list[L][j]-Q_list[L-1][-1])
                costs += m/2
                
        '''
        Step 3
        '''
        var_sum = 0
        for j in range(level):
            m = (2**j)*M
            var_sum += math.sqrt(np.var(Y_list[j])*m)
            
        for j in range(level):
            var = np.var(Y_list[j])
            
            m = (2**j)*M
            c = 2*var_sum/(eps**2)
            N_L = c*math.sqrt(var/m)
            
            #number of samples should not be decreased (we have drawn these samples already)
            if j==0:
                N[j] = max(N[j],int(N_L))
            else:
                N[j] = min(N[j-1],N[j])
                N[j] = max(N[j],int(N_L))
        print(N)
        
        '''
        Step 4
        Evaluate extra samples at each level as needed for the new N_l.
        '''  
        #Calculate Q on finest mesh
        m = (2**(L))*M 
        theta, b = eigenpairs_discrete(max(m_KL,m),0.3,1)
        for i in range(initial_number,N[level-1]):
            sample = draw_sample(theta,b,func_k,m,m_KL)[0]
            k_list[L].append(sample)
            #k_list.append(sample)
            Q_list[L].append(Q(sample,solve_pde(sample,m,p_0,p_1),p_1))
            costs += m
                  
        #go through all meshs
        for i in range(L-1,-1,-1):
            m = int((2**i)*M)
            for j in range(len(Q_list[i]),N[i+1]):
                sample = convert_sample(k_list[i+1][j],m)
                k_list[i].append(k_list[i+1][j])
                Q_list[i].append(Q(sample,solve_pde(sample,m,p_0,p_1),p_1))
                Y_list[i+1].append(Q_list[i+1][j]-Q_list[i][-1])
                costs += m
            
            for j in range(len(Q_list[i]),N[i]):
                sample = draw_sample(theta,b,func_k,m,m_KL)[0]
                k_list[i].append(sample)
                #k_list.append(sample)
                Q_list[i].append(Q(sample,solve_pde(sample,m,p_0,p_1),p_1))
                costs += m
                        
        #Calculate Y_0 on the coarsest mesh
        for i in range(len(Y_list[0]),N[0]):
            Y_list[0].append(Q_list[0][i])
           
        '''
        Step 5
        If L greater or equal 1, test for convergence.
        '''
        estimator = 0
        for j in range(len(Y_list)):
            estimator += sum(Y_list[j])/len(Y_list[j])
        
        if L>=1:
            #print(abs(sum(Y_list[-1])/len(Y_list[-1]) - sum(Y_list[-2])/len(Y_list[-2])))
            if(max(abs(sum(Y_list[-1])/len(Y_list[-1])),abs(sum(Y_list[-2])/len(Y_list[-2]))) < M**(-alpha)):
                break
            
        if L>=5:
            break
        '''
        Step 6
        Set L=L+1 and restart the algorithm.
        '''
        L = L + 1
        
    print(f"Kosten: {costs}")
    print(estimator)
    return estimator
