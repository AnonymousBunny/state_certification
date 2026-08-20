import numpy as np
import json
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import time
import torch.nn.functional as F
from math import sqrt, pi
atol = 1e-13

def get_complex (r, theta):
    return r*np.exp(1j*theta)



def gen_array(n, arr= None, device= "cpu"):
    if (arr==None):
        arr= torch.randint(low=0, high=4, size=(n,), device= device)
    states = torch.tensor([
        [0, 1],
        [sqrt(1 / 3), sqrt(2 / 3)],
        [sqrt(1 / 3), get_complex(sqrt(2 / 3), 2 * pi / 3)],
        [sqrt(1 / 3), get_complex(sqrt(2 / 3), 4 * pi / 3)],
    ], device= device, dtype= torch.cfloat)
    coeffs= torch.tensor([1], device=device, dtype= torch.cfloat)
    n= len(arr)
    for j in range(n):
        coeffs= torch.kron(coeffs, states[arr[j]])
    return coeffs.view((1<<n))







def memory_efficient_generate_batch_GHO_state(b, n, r, k, indices_1= None, indices_2= None, device= "cpu"):
    N= (1<<n)
    R= (1<<r)
    M= (1<<(n-r))

    f= torch.zeros(b,R, M, device= device, dtype= torch.cfloat)
    for batch_id in range(b):
        for i in range(k):
            new_term= torch.outer(gen_array(r, indices_1[batch_id][i], device= device), gen_array(n-r, indices_2[batch_id][i], device= device))
 
            f[batch_id]+=new_term

    flattened_f= f.view(b, R*M) #(b, R*M)
    normalized_f= F.normalize(flattened_f, dim=1).view(b, R, M) #(b, R, M)

    return normalized_f





class GHO_experiment():
    def __init__ (self, batch_size, n, r, k, target_indices, lab_indices, device= "cpu"):
        self.n= n
        self.r= r
        self.k=k
        self.b= batch_size
        self.device= device
        N= (1<<n)
        R= (1<<r)
        M= (1<<(n-r))
        target_indices_1= target_indices[:, :, :r]
        target_indices_2= target_indices[:, :, r:]
        lab_indices_1= lab_indices[:, :, :r]
        lab_indices_2= lab_indices[:, :,  r:]
        x= memory_efficient_generate_batch_GHO_state(self.b, self.n, self.r, self.k, target_indices_1, target_indices_2, device= self.device)
        w= memory_efficient_generate_batch_GHO_state(self.b, self.n, self.r, self.k, lab_indices_1, lab_indices_2, device= self.device)
        self.f= x #b, R, M
        self.row_normalized_f= F.normalize(self.f, dim=2)


        self.g= torch.conj(w)

    def fidelity(self):
        output= torch.abs(torch.einsum('ijk,ijk->i', self.f, self.g)) # (b)
        return output*output

    def proxy_fidelity (self):
        inner_products = torch.einsum('ijk,ijk->ij', self.row_normalized_f, self.g) #(b, R)
        fidelities= torch.abs(inner_products) #(b,R)
        fidelities= fidelities*fidelities
        proxy_fidelities= torch.sum(fidelities, dim=1) #b
        return proxy_fidelities #b

    @classmethod
    @torch.inference_mode()
    def run_batched_experiment(cls, batch_size, n, r, k, target_indices, lab_indices,
                               test_id= None, print_flag=0,
                               device= "cpu", output_path= None):
        if (test_id is None):
            test_id= f'Experiment with n={n}, r={r}, k={k}, batch size = {batch_size}'
        if (print_flag!=0):
            print (test_id)
        start_time= time.perf_counter()
        gho_states= cls(batch_size, n, r, k, target_indices, lab_indices, device= device)
        fidelities= gho_states.fidelity()
        proxy_fidelities= gho_states.proxy_fidelity()
        deltas= proxy_fidelities - fidelities
        output_list= []
        list_fid= fidelities.detach().cpu().tolist()
        list_prox_fid= proxy_fidelities.detach().cpu().tolist()
        list_deltas= deltas.detach().cpu().tolist()
        if (batch_size==1):
            return (list_fid[0], list_prox_fid[0])
        for i in range(len(list_fid)):
            output_list.append((list_fid[i], list_prox_fid[i], list_deltas[i]))
        if (output_path is not None):
            with open(output_path, 'w') as file:
                json.dump(output_list, file)
        if (print_flag==1):
            print ("Number of experiments = ", batch_size)
            print ("Average fidelity = ", fidelities.mean().item())
            print ("Average proxy fidelity = ", proxy_fidelities.mean().item())
            print ("Average difference = ", deltas.mean().item())
            print ("Standard deviation of proxy fidelity = ", proxy_fidelities.std().item())
            print (f'Time taken = {time.perf_counter()-start_time} seconds')
        return proxy_fidelities.mean().item()

 


def run_experiment (target_indices, lab_indices, b, k, device= "cpu"): #target_indices: (b*k, n), #lab_indices: (b*k, n)
    n= target_indices.shape[-1]
    proxy_fids= []
    for r in range(1, n):
        print ("doing ", r)
        cur_time= time.perf_counter()
        g= GHO_experiment(b, n, r, k, target_indices, lab_indices, device= device)
        prox_fid= GHO_experiment.run_batched_experiment(b, n, r, k, target_indices, lab_indices, device= device)
        print ("output : ", prox_fid)
        next_time= time.perf_counter()
        print ("time taken = ", next_time - cur_time)
        
        proxy_fids.append(prox_fid)

    return proxy_fids

def run(n, b, k, device= "cpu"):
    target_indices= torch.randint(low=0, high=4, size= (b, k, n,), device= device)  
    lab_indices= torch.randint(low=0, high=4, size= (b,k, n,), device= device)
    proxy_fids= run_experiment(target_indices, lab_indices, b, k, device = device)
    x_array= [r for r in range(1, n)]
    plt.plot(x_array, proxy_fids)
    plt.xlabel("number of measured qubits")
    plt.ylabel("proxy fidelity")
    plt.legend()
    plt.show()
    caption=f'Number of qubits = {n}, Number of states in superposition = {k}, Batch size = {b}'
    print (caption)

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n=20
    batch_size= 20
    k= 100

    run(n, batch_size, k)
