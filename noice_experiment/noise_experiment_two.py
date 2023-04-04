import torch
from fugw.mappings import FUGWBarycenter
from sklearn.gaussian_process.kernels import Matern
import numpy as np
import matplotlib.pyplot as plt

from noice_experiment.testeroni import FeketeGrid, diag_var_process, ar_coeff

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

import noice_experiment.utils as utils
from fugw.solvers import FUGWSolver, FUGWSparseSolver

################
# Experiment 2: Moritz_Model
################
# copy the class FeketeGrid(BaseGrid) from climnet.grids
# get lon: list of longitude values, lat: list of latitude values
# generate a Fakete-grid and save it as m times m matrix as numpy array

m = 50

fekete_grid = FeketeGrid(m)
grid = fekete_grid.grid

cartesian_grid, source_embeddings = utils.spherical2distance(grid['lon'], grid['lat'])
kernel = 1.0 * Matern(length_scale=0.2, nu=0.5)
cartesian_grid = np.array(cartesian_grid[0][:][:]).T

cov = kernel(cartesian_grid)

data = []
weights_list = []
geometry_list = []

print(np.shape(source_embeddings))


for irun in range(4):
    seed = irun
    print(seed)
    np.random.seed(seed)
    F = diag_var_process(ar_coeff, cov, 1)

    Ds =  torch.Tensor(source_embeddings).to(device)
    data.append(F)
    geometry_list.append(Ds)
    weights_list.append(torch.Tensor(np.ones(m)/m).to(device))

fugw_barycenter = FUGWBarycenter()
_, features, geometry, plans, _, loss = fugw_barycenter.fit(weights_list,
                        data,
                        geometry_list,
                        barycenter_size=50,
                        solver="sinkhorn",
                        device=device,
                        solver_params={'ibpp_eps_base':1e4}
                        )

'''barycenter_weights: np.array of size (barycenter_size)
        barycenter_features: np.array of size (barycenter_size, n_features)
        barycenter_geometry: np.array of size
            (barycenter_size, barycenter_size)
        plans: list of arrays
        duals: list of (array, array)
        losses_each_bar_step:'''

'''fugw = FUGWSolver(
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


source_embeddings = torch.Tensor(cartesian_grid).to(device)
target_embeddings = torch.Tensor(cartesian_grid).to(device)

F = torch.cdist(torch.Tensor(data[0]).to(device), torch.Tensor(data[1]).to(device))
Ds = torch.cdist(source_embeddings, source_embeddings)
print(Ds)
Dt = torch.cdist(target_embeddings, target_embeddings)
print(Dt)
print(F)

Ds_normalized = Ds / Ds.max()
Dt_normalized = Dt / Dt.max()

res = fugw.solve(
    alpha=0.8,
    rho_s=2,
    rho_t=3,
    eps=0.02,
    reg_mode="independent",
    F=F,
    Ds=Ds,
    Dt=Dt,
    init_plan=None,
    solver='mm',
    verbose=True,
)

pi = res["pi"]
print(pi)
plt.title("Optimal Coupling")
plt.imshow(pi)
plt.colorbar()
plt.show()

plt.title("Coupling Result")
plt.imshow(Ds@pi)
plt.colorbar()
plt.show()'''

'''fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cartesian_grid[:,0], cartesian_grid[:,1], cartesian_grid[:,2], c='r', marker='o')
for i in range(m):
    for j in range(m):
        ax.plot([cartesian_grid[i,0], cartesian_grid[j,0]], [cartesian_grid[i,1], cartesian_grid[j,1]], [cartesian_grid[i,2], cartesian_grid[j,2]], c='b')
plt.show()'''

'''plt.title("start distance matrix")
plt.imshow(Ds)
plt.colorbar()
plt.show()

gamma = res["gamma"]
duals_pi = res["duals_pi"]
duals_gamma = res["duals_gamma"]
loss_steps = res["loss_steps"]
loss = res["loss"]
loss_entropic = res["loss_entropic"]
loss_times = res["loss_times"]'''

################
# Experiment 2: Moritz_Model
################
