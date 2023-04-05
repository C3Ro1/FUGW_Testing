import numpy
import torch
from fugw.mappings import FUGWBarycenter
from sklearn.gaussian_process.kernels import Matern
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

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


data = []
weights_list = []
geometry_list = []
noise_in_features = True

for _ in range(1):
    if noise_in_features:
        for noise_level in numpy.linspace(0.01,0.99,15):
            for irun in range(5):
                kernel = 1.0 * Matern(length_scale=0.2, nu=noise_level)
                cov = kernel(cartesian_grid)
                print(cov)
                seed = irun
                print(seed)
                np.random.seed(seed)
                F = torch.Tensor(diag_var_process(ar_coeff, cov, 1)).to(device)
                Ds = torch.Tensor(source_embeddings).to(device)
                data.append(F)
                geometry_list.append(Ds)
                weights_list.append(torch.Tensor(np.ones(m) / m).to(device))
                pass
            fugw_barycenter = FUGWBarycenter(alpha=0.5, force_psd=False, learn_geometry=True)
            weights, features, geometry, plans, duals, loss = \
                fugw_barycenter.fit(weights_list,
                                    data,
                                    geometry_list,
                                    barycenter_size=50,
                                    solver="sinkhorn",
                                    device=device,
                                    nits_barycenter=10
                                    )

            # 3d plot for the cartesian grid showing the positions x y and z in space
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(cartesian_grid[:, 0], cartesian_grid[:, 1], cartesian_grid[:, 2], c='r', marker='o')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.show()

            geometry_grid = geometry.cpu().detach().numpy()
            embedding = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
            geometry_grid = embedding.fit_transform(geometry_grid)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(geometry_grid[:, 0], geometry_grid[:, 1], geometry_grid[:, 2], c='r', marker='o')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.show()

            fugw = FUGWSolver(
                nits_bcd=100,
                nits_uot=1000,
                tol_bcd=1e-7,
                tol_uot=1e-7,
                early_stopping_threshold=1e-5,
                eval_bcd=2,
                eval_uot=10,
                ibpp_eps_base=1e3,
            )

            F = torch.cdist(torch.Tensor(np.ones((m * m, 1))),
                            torch.Tensor(np.ones((m * m, 1))))

            res = fugw.solve(
                alpha=0.5,
                rho_s=2,
                rho_t=3,
                eps=0.02,
                reg_mode="independent",
                F=F,
                Ds=geometry,
                Dt=Ds,
                init_plan=None,
                solver="mm",
                verbose=True,
            )
            loss = res['loss_entropic'][-1]
            pi = res['pi']

            print(loss)
            plt.imshow(pi)
            plt.colorbar()
            plt.show()
        pass

    else:
        for noice_type in utils.noice_types:
            F = torch.Tensor(np.ones((1,50))).to(device)
            Ds = torch.Tensor(noice_type(source_embeddings)).to(device)
            data.append(F)
            geometry_list.append(Ds)
            weights_list.append(torch.Tensor(np.ones(m) / m).to(device))
        pass
    pass
'''
fugw_barycenter = FUGWBarycenter(alpha=0.5, force_psd=False, learn_geometry=True)
weights, features, geometry, plans, duals, loss = \
    fugw_barycenter.fit(weights_list,
                        data,
                        geometry_list,
                        barycenter_size=50,
                        solver="sinkhorn",
                        device=device,
                        nits_barycenter=10
                        )

# 3d plot for the cartesian grid showing the positions x y and z in space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cartesian_grid[:, 0], cartesian_grid[:, 1], cartesian_grid[:, 2], c='r', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

geometry_grid = geometry.cpu().detach().numpy()
embedding = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
geometry_grid = embedding.fit_transform(geometry_grid)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(geometry_grid[:, 0], geometry_grid[:, 1], geometry_grid[:, 2], c='r', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

fugw = FUGWSolver(
    nits_bcd=100,
    nits_uot=1000,
    tol_bcd=1e-7,
    tol_uot=1e-7,
    early_stopping_threshold=1e-5,
    eval_bcd=2,
    eval_uot=10,
    ibpp_eps_base=1e3,
)


F = torch.cdist(torch.Tensor(np.ones((m*m, 1))),
                torch.Tensor(np.ones((m*m, 1))))


res = fugw.solve(
    alpha=0.5,
    rho_s=2,
    rho_t=3,
    eps=0.02,
    reg_mode="independent",
    F=F,
    Ds=geometry,
    Dt=Ds,
    init_plan=None,
    solver="mm",
    verbose=True,
    )
loss = res['loss_entropic'][-1]
pi = res['pi']

print(loss)
plt.imshow(pi)
plt.colorbar()
plt.show()
'''

################
# Experiment 2: Moritz_Model
################
