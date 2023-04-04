from datetime import time

import torch
from sklearn.gaussian_process.kernels import Matern
from torch import ones
import numpy as np
import matplotlib.pyplot as plt

from testeroni import FeketeGrid, diag_var_process, ar_coeff

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

import utils
from stochatic_block_model import StochasticBlock
from fugw.solvers import FUGWSolver, FUGWSparseSolver

################
#Experiment 1: Stochastic_Block_Model
################
number_of_nodes_s = 3
number_of_nodes_t = 3

number_of_blocks_s = 3
number_of_blocks_t = 3

Ds = []
D_t = None

for seed in range(10, 100, 10):
    for p in np.linspace(0.5, 0.9, 12):

        stochastic_block = StochasticBlock(number_of_blocks=number_of_blocks_s,
                                           number_of_nodes=number_of_nodes_s,
                                           p=[p], seed=seed)
        stochastic_block.generate_model()
        Converter = utils.AdjGeodConverter(stochastic_block.model[0])
        D_s, adj_D_s, weights = Converter.generate_geodesic()
        Ds.append(torch.Tensor(np.multiply(D_s, D_s)).to(device))
    stochastic_block = StochasticBlock(number_of_blocks=number_of_blocks_t,
                                       number_of_nodes=number_of_nodes_t,
                                       p=[0.9], seed=seed)
    stochastic_block.generate_model()
    Converter = utils.AdjGeodConverter(stochastic_block.model[0])
    D_t, adj_D_t, _ = Converter.generate_geodesic()

if D_t is not None:
        plt.title("Target Distance Matrix")
        plt.imshow(D_t)
        plt.colorbar()
        plt.show()

init_plan = (
    (ones(number_of_nodes_s,
          number_of_nodes_s) / number_of_nodes_s * number_of_nodes_t * number_of_blocks_s * number_of_blocks_t).to(
        device)
)

fugw = FUGWSolver(
    nits_bcd=100,
    nits_uot=1000,
    tol_bcd=1e-7,
    tol_uot=1e-7,
    early_stopping_threshold=1e-5,
    eval_bcd=2,
    eval_uot=10,
    # Set a high value of ibpp, otherwise nans appear in coupling.
    # This will generally increase the computed fugw loss.
    ibpp_eps_base=1e3,
)

DT = torch.Tensor(np.multiply(D_t, D_t)).to(device)
F = torch.cdist(torch.Tensor(np.ones((number_of_nodes_s*number_of_blocks_s, 2048))),
                torch.Tensor(np.ones((number_of_nodes_t*number_of_blocks_t, 2048))))

out_loss = []
out_pi = []

for DS in Ds:
    for alpha in np.linspace(0.1, 0.9, 7):
        res = fugw.solve(
            alpha=alpha,
            rho_s=2,
            rho_t=3,
            eps=0.02,
            reg_mode="independent",
            F=F,
            Ds=DS,
            Dt=DT,
            init_plan=torch.Tensor(np.ones((number_of_nodes_s*number_of_blocks_s,
                                            number_of_nodes_t*number_of_blocks_t))),
            solver="mm",
            verbose=True,
        )
        out_loss.append(res['loss_entropic'][-1])
        out_pi.append(res['pi'])
        pass

'''pi = res["pi"]
plt.imshow(pi)
plt.colorbar()
plt.show()'''

'''gamma = res["gamma"]
plt.imshow(gamma)
plt.colorbar()
plt.show()'''

duals_pi = res["duals_pi"]
print(duals_pi)
duals_gamma = res["duals_gamma"]
print(duals_gamma)
loss_steps = res["loss_steps"]
print(loss_steps)
loss = res["loss"]
print(loss)
loss_entropic = res["loss_entropic"]
print(loss_entropic)
loss_times = res["loss_times"]
print(loss_times)

################
#Experiment 1: Stochastic_Block_Model
################
