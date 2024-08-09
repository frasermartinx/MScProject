from symengine import symbols, function_symbol, DenseMatrix, sin, cos, exp, Piecewise
import numpy as np
import torch
import multiprocessing
from sympy import  lambdify, besselj
from scipy.special import jn_zeros



def generate_2d(K,R_lim):
    r,theta = symbols("r theta")
    u = function_symbol("u",r,theta)
    u = 0*r +0
    for n in range(1,K):
        for k in range(1,K):
            alpha = jn_zeros(n,k)[k-1]
            eigenvalue = alpha**2 / R_lim**2
            u = u + (np.random.normal(0,1/np.sqrt(n+k))*besselj(n,r*alpha/R_lim)*cos(n*theta))/np.sqrt(eigenvalue)+ (np.random.normal(0,1/np.sqrt(n+k))*besselj(n,r*alpha/R_lim)*sin(n*theta))/np.sqrt(eigenvalue)
    return u

def get_function(boundary_condition, K):
    u = None
    if boundary_condition.lower() =="dirichlet":
        u = generate_sine_2d(K)
    elif boundary_condition.lower() =="neumann":
        u = generate_cosine_2d(K)
    else:
        raise TypeError("Only Dirichlet or Neumann boundary conditions.")
    return u

def divergence_A_nablau(u,elliptic_matrix):
    r,theta = symbols("r theta")
    divA_nablau = function_symbol("divA_nablau", r,theta)

    a = elliptic_matrix[0]
    b = elliptic_matrix[1]
    c = elliptic_matrix[2]
    d = elliptic_matrix[3]

    a_r = function_symbol("a_r", r,theta)
    b_r = function_symbol("b_r", r,theta)
    c_theta = function_symbol("c_theta", r,theta)
    d_theta = function_symbol("d_theta", r,theta)   


    a_r = a.diff(r)
    b_r = b.diff(r)
    c_theta = c.diff(theta)
    d_theta = d.diff(theta)

    lin = a_r * u.diff(r) + a * u.diff(r).diff(r)
    rinv =  a * u.diff(r) + b_r * u.diff(theta) + b*u.diff(r).diff(theta) + c_theta*u.diff(r) + c * u.diff(r).diff(theta)
    rinv = Piecewise((1/r * rinv, r!=0), (0, True))


    rdinv = d_theta * u.diff(theta) + d * u.diff(theta).diff(theta)
    rdinv = Piecewise((1/r**2 * rdinv, r!=0), (0, True))


    divA_nablau = lin + rinv + rdinv
    return (-1)*divA_nablau 


def generate_data(dimension, grid_size, truncation_order, elliptic_matrix, nonlinear_term, boundary_condition,idx):
    R_lim = 1
    r, theta = symbols("r theta")
    f = function_symbol("f",r,theta)
    w = symbols("w")
    K = np.random.randint(2,truncation_order+1)
    u = generate_2d(K, R_lim)
    c_u = nonlinear_term.subs({w: u})

    f = divergence_A_nablau(u,elliptic_matrix) + c_u

    
    cor = np.linspace(1e-8,R_lim,grid_size)
    cor_theta = np.linspace(1e-8,2*np.pi,grid_size)
    R, Theta = np.meshgrid(cor,cor_theta)
    

    R_flat = R.ravel()
    Theta_flat = Theta.ravel()

    f_func = lambdify((r, theta), f)
    u_func = lambdify((r, theta), u)

    f_flat = f_func(R_flat, Theta_flat)
    u_flat = u_func(R_flat, Theta_flat)

    f_reshaped = np.array(f_flat).reshape(R.shape)
    u_reshaped = np.array(u_flat).reshape(R.shape)

    input_data = torch.tensor(f_reshaped)
    output_data = torch.tensor(u_reshaped)

    input_data = input_data[None,:]
    output_data = output_data[None,:]
    

    print("progress:", idx)
    

    return input_data, output_data, idx


def generate_and_enqueue_data(x_data: torch.Tensor, y_data: torch.Tensor, dimension: int, grid_size: int, 
                              truncation_order: int, elliptic_matrix, nonlinear_term, boundary_condition, idx: int):
    
    result1, result2, idx = generate_data(dimension, grid_size, truncation_order, elliptic_matrix, nonlinear_term, boundary_condition, idx)
    x_data[idx,:,:,:] = result1
    y_data[idx,:,:] = result2

def save_data_in_parallel(length: int, dimension: int, grid_size: int, truncation_order: int, elliptic_matrix,
                           nonlinear_term, boundary_condition, x_path, y_path):
    multiprocessing.set_start_method('spawn')
    grid_size = grid_size
    length = length
    input_functions = 1
    x_data = torch.zeros((length,input_functions,grid_size, grid_size))
    y_data = torch.zeros((length,grid_size,grid_size))
    x_data.share_memory_()
    y_data.share_memory_()

    args = [(x_data, y_data, dimension, grid_size, truncation_order, elliptic_matrix,nonlinear_term,
             boundary_condition, idx) for idx in range(length)]
    # torch.set_num_threads(1)
    with multiprocessing.Pool() as pool:
        pool.starmap(generate_and_enqueue_data, args)

    torch.save(x_data, x_path)
    torch.save(y_data, y_path)




if __name__ == '__main__':
    r,theta = symbols("r theta")
    a = function_symbol("a", r,theta)
    b = function_symbol("b", r,theta)
    c = function_symbol("c", r,theta)
    d = function_symbol("d", r,theta)

    w = symbols("w")
    c_u = function_symbol("c_u",w)

    a = 1 + 0*r + 0*theta
    b = 0 + 0*r + 0*theta
    c = 0 + 0*r + 0*theta
    d = 1 + 0*r + 0*theta
    c_u = w**2
    length = 200

    save_data_in_parallel(
        length = length,
        dimension= 2,
        grid_size= 85,
        truncation_order = 20,
        elliptic_matrix= [a,b,c,d],
        nonlinear_term= c_u,
        boundary_condition= "dirichlet",
        x_path= f'cosine_1_to_20_positive_div_particularA_norm_{length}_x_test.pt',
        y_path= f'cosine_1_to_20_positive_div_particularA_norm_{length}_y_test.pt'
    )
    