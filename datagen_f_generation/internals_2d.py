from symengine import symbols, function_symbol, DenseMatrix, sin, cos, exp
import numpy as np
import torch
import multiprocessing
from sympy import  lambdify



def generate_F(K):
    x,y = symbols("x y")
    f = function_symbol("u",x,y)
    f = 0*x +0
    for i in range(1,K):
        for j in range(1,K):
            f = f + np.random.normal(0,1/np.sqrt(i+j))*cos(np.pi*i*x)*cos(np.pi*j*y)/np.sqrt(((np.pi*i)**2+(np.pi*j)**2))+ np.random.normal(0,1/np.sqrt(i+j))*sin(np.pi*i*x)*sin(np.pi*j*y)/np.sqrt(((np.pi*i)**2+(np.pi*j)**2))
    return f



def generate_data(dimension, grid_size, truncation_order, idx):
    
    x, y = symbols("x,y")
    K = np.random.randint(2,truncation_order+1)
    f = generate_F(K)
    


    cor = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(cor, cor)

    X_flat = X.ravel()
    Y_flat = Y.ravel()

    f_func = lambdify((x, y), f)

    f_flat = f_func(X_flat, Y_flat)

    f_reshaped = np.array(f_flat).reshape(X.shape)

    input_data = torch.tensor(f_reshaped)

    input_data = input_data[None,:]
    

    print("progress:", idx)
    

    return input_data, idx


def generate_and_enqueue_data(x_data: torch.Tensor, dimension: int, grid_size: int, 
                              truncation_order: int, idx: int):
    
    result, idx = generate_data(dimension, grid_size, truncation_order,idx)
    x_data[idx,:,:,:] = result

def save_data_in_parallel(length: int, dimension: int, grid_size: int, truncation_order: int, x_path):
    multiprocessing.set_start_method('spawn')
    grid_size = grid_size
    length = length
    input_functions = 1
    x_data = torch.zeros((length,input_functions,grid_size, grid_size))
    x_data.share_memory_()

    args = [(x_data, dimension, grid_size, truncation_order, idx) for idx in range(length)]
    # torch.set_num_threads(1)
    with multiprocessing.Pool() as pool:
        pool.starmap(generate_and_enqueue_data, args)

    torch.save(x_data, x_path)




if __name__ == '__main__':
    x,y = symbols("x y")
    a = function_symbol("a", x,y)
    b = function_symbol("b", x,y)
    c = function_symbol("c", x,y)
    d = function_symbol("d", x,y)

    w = symbols("w")
    c_u = function_symbol("c_u",w)

    a = 1 + 0*x + 0*y
    b = 0 + 0*x + 0*y
    c = 0 + 0*x + 0*y
    d = 1 + 0*x + 0*y
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
    