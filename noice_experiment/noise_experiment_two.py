from datetime import time
import torch
from sklearn.gaussian_process.kernels import Matern
import numpy as np
import matplotlib.pyplot as plt

import extensive_testing
from extensive_testing.testeroni import FeketeGrid, diag_var_process, ar_coeff
from fugw.mappings import FUGWBarycenter

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

from extensive_testing.experimental_environment.noice_experiment import utils
from fugw.solvers import FUGWSolver, FUGWSparseSolver

Ds = []
D_t = None


################
#Experiment 2: Moritz_Model
################
# copy the class FeketeGrid(BaseGrid) from climnet.grids
# get lon: list of longitude values, lat: list of latitude values

#generate a Fakete-grid and save it as m times m matrix as numpy array

m =  50

fekete_grid = FeketeGrid(m)
a  = fekete_grid.grid

cartesian_grid = extensive_testing.experimental_environment.noice_experiment.utils.spherical2cartesian(a['lon'],a['lat'])
kernel = 1.0 * Matern(length_scale=0.2, nu=0.5)
#cartasian grid from tupel to numpy array
cartesian_grid = np.array(cartesian_grid).T

cov = kernel(cartesian_grid)



data = []
for irun in range(2):
    seed = int(time().second)
    np.random.seed(seed)
    data.append(diag_var_process(ar_coeff, cov, 100))



fugw_barycenter = FUGWBarycenter()
fugw_barycenter.fit(
    weights_list, features_list, geometry_list, device=device
)


print(cartesian_grid)

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

ns = 100
ds = 3
nt = 100
dt = 3
nf = 3

source_features = torch.rand(ns, nf).to(device)
target_features = torch.zeros(nt, nf).to(device)
source_embeddings = torch.rand(ns, ds).to(device)
target_embeddings = torch.rand(nt, dt).to(device)

F = torch.cdist(torch.Tensor(data).to(device), target_features)
Ds = torch.cdist(source_embeddings, source_embeddings)
Dt = torch.cdist(target_embeddings, target_embeddings)

Ds_normalized = Ds / Ds.max()
Dt_normalized = Dt / Dt.max()

res = fugw.solve(
    alpha=0.8,
    rho_s=2,
    rho_t=3,
    eps=0.02,
    reg_mode="independent",
    F=F,
    Ds=Ds_normalized,
    Dt=Dt_normalized,
    init_plan=None,
    solver='mm',
    verbose=True,
)

pi = res["pi"]
plt.title("Optimal Coupling")
plt.imshow(pi)
plt.colorbar()
plt.show()

gamma = res["gamma"]
duals_pi = res["duals_pi"]
duals_gamma = res["duals_gamma"]
loss_steps = res["loss_steps"]
loss = res["loss"]
loss_entropic = res["loss_entropic"]
loss_times = res["loss_times"]


'''
# now data is a n_time x n_gridpoints matrix of the time series you generated. FUGW should recover the original distance matrix D_ij. Test tha


################
#Experiment 2: Moritz_Model
################

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
F = torch.cdist(torch.Tensor(np.ones((number_of_nodes_s*number_of_blocks_s, 1))),
                torch.Tensor(np.ones((number_of_nodes_t*number_of_blocks_t, 1))))

out_loss_1 = []
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
        out_loss_1.append(res['loss_entropic'][-1])
        out_pi.append(res['pi'])
        pass

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
'''