import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({'text.latex.preamble':[r'\usepackage{physics}']})
plt.rcParams.update({'text.latex.preamble':[r'\usepackage{amsmath}']})
plt.style.use('seaborn-whitegrid')

# Function to update velocity
def new_v(v,alpha,n):
    return v - alpha*v + (2*np.random.randint(0,2,n)-1)

# Function that computes the expectation value of v_t v_t' for |t-t'| going
# from 0 to T
def sim(t_0,T,alpha,n):
    # Array to hold the expectation values of v_t v_t'
    results = []
    # Array to store the velocity of all n runs
    v = np.zeros((n))
    for i in range(t_0):
        v = new_v(v,alpha,n)
    v_t = v
    for i in range(T+1):
        v = v_t
        for j in range(i):
            v = new_v(v,alpha,n)
        results.append(np.mean(v*v_t))
    return results

# Function that will be given as input to scipy curve_fit
def f(x,A,B):
    return A*np.exp(-B*x)

# Function that takes in results of a simulation and fits a curve to them
def fit_params(results):
    A,B = fit(f,np.arange(0,len(results)),results)[0]
    return A,B

# Function that makes the plot of the expectation value of v_t v_t' for |t-t'|
# going from 0 to T, along with the best fit curve
def plot(results,A,B):
    x = np.arange(0,len(results))
    plt.plot(x,results,color='green',label='simulation',lw=0,marker='.',ms=4)
    plt.plot(x,f(x,A,B),color='black',label='fit')
    plt.xlabel('$|t-t\'|$')
    plt.ylabel('$\\langle v_t v_{t\'} \\rangle$')
    plt.title('Simulation and curve fit with A={}, B={}'.format(np.round(A,1),np.round(B,3)))
    plt.legend(frameon=True,shadow=True)
    plt.show()

# Function to graph best fit parameters A and B as a function of alpha. Returns
# two arrays containing the values of A and B for all values of alpha given as
# input
def parameters_vs_alpha(alpha_array,t_0,T,n):
    A_array = []
    B_array = []
    for alpha in alpha_array:
        results = sim(t_0,T,alpha,n)
        A,B = fit_params(results)
        A_array.append(A)
        B_array.append(B)
    return A_array, B_array

# Function that plots the best fit parameters A and B versus alpha
def plot_AB(alpha_array,A_array,B_array):
    plt.plot(alpha_array,A_array,color='green')
    plt.xlabel('$\\alpha$')
    plt.ylabel('$A$')
    plt.title('Dependence of $A$ on $\\alpha$: $A \propto \\frac{1}{\\alpha}$')
    plt.show()
    plt.plot(alpha_array,B_array,color='green')
    plt.xlabel('$\\alpha$')
    plt.ylabel('$B$')
    plt.title('Dependence of $B$ on $\\alpha$: $B \propto \\alpha$')
    plt.show()

################################################################################
# Now just using the functions to solve the problem
################################################################################
# Running a simulation for alpha = 0.02 and ploting the results
results = sim(50000, 400, 0.02, 100000)
A,B = fit_params(results)
plot(results,A,B)

# Findinf the dependence of A and B on alpha
alpha_array = np.linspace(0.02,0.1,100)
A_array, B_array = parameters_vs_alpha(alpha_array,1000,400,10000)
plot_AB(alpha_array,A_array,B_array)
