# copy the class FeketeGrid(BaseGrid) in climnet.grids
# get lon: list of longitude values, lat: list of latitude values


import math
from warnings import WarningMessage
import numpy as np
import matplotlib.pyplot as plt

from numpy.fft import fft
import time
from scipy.spatial import SphericalVoronoi
import os
from scipy.spatial.transform import Rotation as Rot
import warnings
import fnmatch, sys

from sklearn.gaussian_process.kernels import Matern

if fnmatch.fnmatch(sys.version, '3.7.1*'):
    import pickle5 as pickle
else:
    import pickle

#! /usr/bin/env python3
"""
fekete -  Estimation of Fekete points on a unit sphere

          This module implements the core algorithm put forward in [1],
          allowing users to estimate the locations of N equidistant points on a
          unit sphere.

[1] Bendito, E., Carmona, A., Encinas, A. M., & Gesto, J. M. Estimation of
    Fekete points (2007), J Comp. Phys. 225, pp 2354--2376  
    https://doi.org/10.1016/j.jcp.2007.03.017

"""
# Created: Sat Jun 19, 2021  06:21pm Last modified: Sat Jun 19, 2021  06:21pm
#
# Copyright (C) 2021  Bedartha Goswami <bedartha.goswami@uni-tuebingen.de> This
# program is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------


import numpy as np
from scipy.spatial.distance import pdist
from tqdm import tqdm
from numba import jit
from scipy.spatial import SphericalVoronoi
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

G = 6.67408 * 1E-11         # m^3 / kg / s^2


def bendito(N=100, a=1., X=None, maxiter=1000, verbose=True):
    """
    Return the Fekete points according to the Bendito et al. (2007) algorithm.

    Parameters
    ----------
    N : int
        Number of points to be distributed on the surface of the unit sphere.
        Default is `N = 100`.
    a : float
        Positive scalar that weights the advance direction in accordance with
        the kernel under consideration and the surface (cf. Eq. 4 and Table 1
        of Bendito et al., 2007). Default is `a = 1` which corresponds to the
        Newtonian kernel.
    X : numpy.nadarray, with shape (N, 3)
        Initial configuration of points. The array consists of N observations
        (rows) of 3-D (x, y, z) locations of the points. If provided, `N` is
        overriden and set to `X.shape[0]`. Default is `None`.
    maxiter : int
        Maximum number of iterations to carry out. Since the error of the
        configuration continues to decrease exponentially after a certain
        number of iterations, a saturation / convergence criterion is not
        implemented. Users are advised to check until the regime of exponential
        decreased is reach by trying out different high values of `maxiter`.
        Default is 1000.
    verbose : bool
        Show progress bar. Default is `True`.

    Returns
    -------
    X_new : numpy.ndarray, with shape (N, 3)
        Final configuration of `N` points on the surface of the sphere after
        `maxiter` iterations. Each row contains the (x, y, z) coordinates of
        the points. If `X` is provided, the `X_new` has the same shape as `X`.
    dq : numpy.ndarray, with shape (maxiter,)
        Maximum disequilibrium degree after each iteration. This is defined as
        the maximum of the modulus of the disequilibrium vectors at each point
        location. Intuitively, this can be understood as a quantity that is
        proportional to the total potential energy of the current configuration
        of points on the sphere's surface.

    """
     # parse inputs
    if X is None or len(X) == 0:
        print("Initial configuration not provided. Generating random one ...")
        X = points_on_sphere(N)         # initial random configuration
    else:
        N = X.shape[0]

    # core loop
    ## intializ parameters
    dq = []
    w = np.zeros(X.shape)
    ## set up progress bar
    pb_fmt = "{desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}"
    pb_desc = "Estimating Fekete points ..."
    ## iterate
    for k in tqdm(range(maxiter), bar_format=pb_fmt, desc=pb_desc,
                  disable=not verbose):

        # Core steps from Bendito et al. (2007), pg 6 bottom
        ## 1.a. Advance direction
        for i in range(len(X)):
            w[i] = descent_direction_i(X, i)

        # 1.b. Error as max_i |w_i|
        mod_w = np.sqrt((w ** 2).sum(axis=1))
        dq.append(np.max(mod_w))

        ## 2.a. Minimum distance between all points
        d = np.min(pdist(X))
        ## 2.b. Calculate x^k_hat = x^k + a * d^{k-1} w^{k-1}
        Xhat = X + a * d * w

        ## 3. New configuration
        X_new = (Xhat.T / np.sqrt((Xhat ** 2).sum(axis=1))).T
        X = X_new

    return X_new, dq


@jit(nopython=True)
def descent_direction_i(X, i):
    """
    Returns the 3D vector for the direction of decreasing energy at point i.

    Parameters
    ----------
    X : numpy.nadarray, with shape (N, 3)
        Current configuration of points. Each row of `X` is the 3D position
        vector for the corresponding point in the current configuration.
    i : int
        Index of the point for which the descent direction is to be estimated.
        The position vector of point `i` is the i-th row of `X`.

    Returns
    -------
    wi : numpy.ndarray, with shape (3,)
         The vector along which the particle at point `i` has to be moved in
         order for the total potential energy of the overall configuration to
         decrease. The vector is estimated as the ratio of the tangential force
         experienced by the particle at `i` to the magnitude of the total force
         experienced by the particle at `i`. The tangential force is calculated
         as the difference between the total force and the component of the
         total force along the (surface) normal direction at `i`.

    """
    xi = X[i]

    # total force at i
    xi_arr = xi.repeat(X.shape[0]).reshape(xi.shape[0], X.shape[0]).T
    diff = xi_arr - X
    j = np.where(np.sum(diff, axis=1) != 0)[0]
    diff_j = diff[j]
    denom = (np.sqrt(np.square(diff_j).sum(axis=1))) ** 3
    numer = (G * diff_j)
    Fi_tot = np.sum((numer.T / denom).T, axis=0)    # gives 3D net force vector

    # direction of descent towards lower energy
    xi_n = xi / np.sqrt(np.square(xi).sum())
    Fi_n = (Fi_tot * xi_n).sum() * xi_n
    Fi_T = Fi_tot - Fi_n
    wi = Fi_T / np.sqrt(np.square(Fi_tot).sum())

    return wi


def points_on_sphere(N, r=1.):
    """
    Returns random points on the surface of a 3D sphere.

    Parameters
    ----------
    N : int
        Number of points to be distributed randomly on sphere's surface
    r : float
        Positive number denoting the radius of the sphere. Default is `r = 1`.

    Returns
    -------
    X : numpy.ndarray, with shape (N, 3)
        Locations of the `N` points on the surface of the sphere of radius `r`.
        The i-th row in `X` is a 3D vector that gives the location of the i-th
        point.
    """
    phi = np.arccos(1. - 2. * np.random.rand(N))
    theta = 2. * np.pi * np.random.rand(N)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np. sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    return np.c_[x, y, z]


def cartesian_to_spherical(X):
    """
    Returns spherical coordinates for a given array of Cartesian coordinates.

    Parameters
    ----------
    X : numpy.ndarray, with shape (N, 3)
        Locations of the `N` points on the surface of the sphere of radius `r`.
        The i-th row in `X` is a 3D vector that gives the location of the i-th
        point.

    Returns
    -------
    theta : numpy.ndaaray, with shape (N,)
        Azimuthal angle of the different points on the sphere. Values are
        between (0, 2pi). In geographical terms, this corresponds to the
        longitude of each location.
    phi : numpy.ndaaray, with shape (N,)
        Polar angle (or inclination) of the different points on the sphere.
        Values are between (0, pi). In geographical terms, this corresponds to
        the latitude of each location.
    r : float
        Radial distance of the points to the center of the sphere. Always
        greater than or equal to zero.
    """
    r = np.sqrt(np.square(X).sum(axis=1))   # radius
    theta = np.arccos(X[:, 2] / r)          # azimuthal angle
    phi = np.arctan(X[:, 1] / X[:, 0])      # polar angle (inclination)

    return theta, phi, r


def spherical_to_cartesian(theta, phi, r=1.):
    """
    Returns Cartesian coordinates for a given array of spherical coordinates.


    Parameters
    ----------
    theta : numpy.ndaaray, with shape (N,)
        Azimuthal angle of the different points on the sphere. Values are
        between (0, 2pi). In geographical terms, this corresponds to the
        longitude of each location.
    phi : numpy.ndaaray, with shape (N,)
        Polar angle (or inclination) of the different points on the sphere.
        Values are between (0, pi). In geographical terms, this corresponds to
        the latitude of each location.
    r : float
        Radial distance of the points to the center of the sphere. Always
        greater than or equal to zero. Default is `r = 1`.

    Returns
    -------
    X : numpy.ndarray, with shape (N, 3)
        Locations of the `N` points on the surface of the sphere of radius `r`.
        The i-th row in `X` is a 3D vector that gives the location of the i-th
        point in `(x, y, z)` coordinates.

    """
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    X = np.c_[x, y, z]

    return X


def plot_spherical_voronoi(X, ax):
    """
    Plot scipy.spatial.SphericalVoronoi output on the surface of a unit sphere.

    Parameters
    ----------
    X : numpy.ndarray, with shape (N, 3)
        Locations of the `N` points on the surface of the sphere of radius `r`.
        The i-th row in `X` is a 3D vector that gives the location of the i-th
        point in `(x, y, z)` coordinates.
    ax : matplotlib.pyplot.Axes
        Axis in which the Voronoi tessellation output is to be plotted.

    Returns
    -------
    ax : matplotlib.pyplot.Axes
        The same axis object used for plotting is returned.

    """
    vor = SphericalVoronoi(X)
    vor.sort_vertices_of_regions()
    verts = vor.vertices
    regs = vor.regions
    for i in range(X.shape[0]):
        verts_reg = np.array([verts[k] for k in regs[i]])
        verts_reg = [list(zip(verts_reg[:, 0], verts_reg[:, 1], verts_reg[:, 2]))]
        ax.add_collection3d(Poly3DCollection(verts_reg,
                                             facecolors="w",
                                             edgecolors="steelblue"
                                             ),
                            )
    ax.set_xlim(-1.01, 1.01)
    ax.set_ylim(-1.01, 1.01)
    ax.set_zlim(-1.01, 1.01)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2],
               marker=".", color="indianred", depthshade=True, s=40)
    return ax



class BaseGrid():
    """Base Grid

    Parameters:
    -----------

    """

    def __init__(self):
        self.grid = None
        return

    def get_distance_equator(self):
        """Return distance between points at the equator."""
        print("Function should be overwritten by subclasses!")
        return None

    def create_grid(self):
        """Create grid."""
        print("Function should be overwritten by subclasses!")
        return None

    def cut_grid(self, lat_range, lon_range):
        """Cut the grid in lat and lon range.

        TODO: allow taking regions around the date line

        Args:
        -----
        lat_range: list
            [min_lon, max_lon]
        lon_range: list
            [min_lon, max_lon]
        """
        if lon_range[0] > lon_range[1]:
            raise ValueError("Ranges around the date line are not yet defined.")
        else:
            print(f"Cut grid in range lat: {lat_range} and lon: {lon_range}")

        idx = np.where((self.grid['lat'] >= lat_range[0])
                       & (self.grid['lat'] <= lat_range[1])
                       & (self.grid['lon'] >= lon_range[0])
                       & (self.grid['lon'] <= lon_range[1]))[0]
        cutted_grid = {'lat': self.grid['lat'][idx], 'lon': self.grid['lon'][idx]}

        return cutted_grid


class FeketeGrid(BaseGrid):
    """Fibonacci sphere creates a equidistance grid on a sphere.

    Parameters:
    -----------
    distance_between_points: float
        Distance between the equidistance grid points in km.
    grid: dict (or 'old' makes old version of fib grid, 'maxmin' to maximize min. min-distance)

        If grid is already computed, e.g. {'lon': [], 'lat': []}. Default: None
    """

    def __init__(self, num_points, num_iter=1000, grid=None, save=True):
        self.distance = get_distance_from_num_points(num_points)
        self.num_points = num_points
        self.num_iter = num_iter
        self.epsilon = None
        self.grid = grid
        if grid is None:  # maxavg is standard
            self.create_grid(num_points, num_iter, save=save)
        self.reduced_grid = None

    def create_grid(self, num_points=1, num_iter=1000, save=True):
        filepath = f'feketegrid_{num_points}_{num_iter}.p'
        if os.path.exists(filepath):
            print(f'\nLoad Fekete grid with {self.num_points} points after {num_iter} iterations.')
            with open(filepath, 'rb') as fp:
                self.grid, self.dq = pickle.load(fp)
                return self.grid
        else:
            print(f'\nCreate Fekete grid with {self.num_points} points after {num_iter} iterations.')
            X, self.dq = bendito(N=num_points, maxiter=num_iter)
            lon, lat = cartesian2spherical(X[:, 0], X[:, 1], X[:, 2])
            self.grid = {'lon': lon, 'lat': lat}
            if save:
                with open(filepath, 'wb') as fp:
                    pickle.dump((self.grid, self.dq), fp, protocol=pickle.HIGHEST_PROTOCOL)
            return self.grid

    def nudge_grid(self, n_iter=1, step=0.01):  # a 100th of a grid_step
        if self.reduced_grid is None:
            raise KeyError('First call keep_original_points')
        leng = len(self.grid['lon'])
        delta = 2 * np.pi * step * self.distance / 6371
        regx, regy, regz = spherical2cartesian(self.reduced_grid['lon'], self.reduced_grid['lat'])
        for iter in range(n_iter):
            perm = np.random.permutation(leng)
            for i in range(leng):
                i = perm[i]
                x, y, z = spherical2cartesian(self.grid['lon'], self.grid['lat'])
                r = np.array([x[i], y[i], z[i]])
                vec2 = np.array([regx[i] - x[i], regy[i] - y[i], regz[i] - z[i]])
                vec2 = vec2 - np.dot(vec2, r) * r
                rot_axis = np.cross(r, vec2)
                rot = Rot.from_rotvec(rot_axis * delta / np.linalg.norm(rot_axis))
                new_grid_cart = rot.as_matrix() @ np.array([x, y, z])
                new_lon, new_lat = cartesian2spherical(new_grid_cart[0, :], new_grid_cart[1, :], new_grid_cart[2, :])
                self.grid = {'lon': new_lon, 'lat': new_lat}

    def get_distance_equator(self):
        """Return distance between points at the equator."""
        return self.distance

    def keep_original_points(self, orig_grid, regular=True):
        if self.grid is None:
            self.create_grid()
        if regular:
            new_lon, used_dists, delete, dists, possible_coords, new_coords = [], [], [], [], [], []
            lons = np.sort(np.unique(orig_grid['lon']))
            lats = np.sort(np.unique(orig_grid['lat']))
            for i in range(len(self.grid['lat'])):
                lo = self.grid['lon'][i]
                la = self.grid['lat'][i]
                pm_lon = np.array([(lons[j] - lo) * (lons[j + 1] - lo) for j in range(len(lons) - 1)])
                pm_lat = np.array([(lats[j] - la) * (lats[j + 1] - la) for j in range(len(lats) - 1)])
                if np.where(pm_lon < 0)[0].shape[0] == 0:
                    rel_lon = [lons[0], lons[-1]]
                else:
                    lon_idx = np.where(pm_lon < 0)[0][0]
                    rel_lon = [lons[lon_idx], lons[lon_idx + 1]]
                if np.where(pm_lat < 0)[0].shape[0] == 0:
                    rel_lat = [lats[0], lats[-1]]
                else:
                    lat_idx = np.where(pm_lat < 0)[0][0]
                    rel_lat = [lats[lat_idx], lats[lat_idx + 1]]
                these_dists = np.array([gdistance((l2, l1), (la, lo)) for l1 in rel_lon for l2 in rel_lat])
                these_coords = np.array([(l1, l2) for l1 in rel_lon for l2 in rel_lat])
                prio = np.argsort(these_dists)
                dists.append(these_dists[prio])
                possible_coords.append(these_coords[prio])

            for idx in np.argsort(np.array(dists)[:, 0]):  # choose the nearest unused neighbor
                i = 0
                while i < 4 and np.any([np.all(possible_coords[idx][i, :] == coord) for coord in new_coords]):
                    if i == 3:
                        delete.append(idx)
                        warnings.warn(
                            f'No neighbors left for  {self.grid["lon"][idx], self.grid["lat"][idx]}. Removing this point.')
                    i += 1
                if i < 4:
                    new_coords.append(possible_coords[idx][i, :])
                    used_dists.append(dists[idx][i])
                # ids = map(id, new_coords)
            dists2 = np.delete(dists, delete, 0)
            new_coords = np.array(new_coords)[np.argsort(np.argsort(np.array(dists2)[:, 0]))]  # inverse permutation
            self.reduced_grid = {'lon': np.array(new_coords)[:, 0], 'lat': np.array(new_coords)[:, 1]}
            self.grid['lon'] = np.delete(self.grid['lon'], delete, 0)
            self.grid['lat'] = np.delete(self.grid['lat'], delete, 0)
            return used_dists
        else:
            raise KeyError('Only regular grids!')

    def min_dists(self, grid2=None):
        if grid2 is None:
            lon1, lon2 = self.grid['lon'], self.grid['lon']
            lat1, lat2 = self.grid['lat'], self.grid['lat']
            d = 9999 * np.ones((len(lon1), len(lon2)))
            for i in range(len(lon1)):
                for j in range((len(lon1))):
                    if i < j:
                        d[i, j] = gdistance((lat1[i], lon1[i]), (lat2[j], lon2[j]))
                    elif i > j:
                        d[i, j] = d[j, i]
            return d.min(axis=1)
        else:  # min dist from self.grid point to other grid
            lon1, lon2 = self.grid['lon'], grid2['lon']
            lat1, lat2 = self.grid['lat'], grid2['lat']
            d = 9999 * np.ones((len(lon1), len(lon2)))
            for i in range(len(lon1)):
                for j in range(len(lon2)):
                    d[i, j] = gdistance((lat1[i], lon1[i]), (lat2[j], lon2[j]))
            return d.min(axis=1)


def get_distance_from_num_points(num_points):
    k = 1 / 2.01155176
    a = np.exp(20.0165958)
    return (a / num_points) ** k


def get_num_points(dist):
    """Relationship between distance and num of points of fibonacci sphere.

    num_points = a*distance**k
    """
    # obtained by log-log fit
    k = -2.01155176
    a = np.exp(20.0165958)
    return int(a * dist ** k)


def maxmin_epsilon(num_points):
    if num_points >= 600000:
        epsilon = 214
    elif num_points >= 400000:
        epsilon = 75
    elif num_points >= 11000:
        epsilon = 27
    elif num_points >= 890:
        epsilon = 10
    elif num_points >= 177:
        epsilon = 3.33
    elif num_points >= 24:
        epsilon = 1.33
    else:
        epsilon = 0.33
    return epsilon


def regular_lon_lat(num_lon, num_lat):  # creates regular grid with borders half the distance of one step at each border
    lon = np.linspace(-180 + 360 / (2 * num_lon), 180 - 360 / (2 * num_lon), num_lon)
    lat = np.linspace(-90 + 180 / (2 * num_lat), 90 - 180 / (2 * num_lat), num_lat)
    return lon, lat


def regular_lon_lat_step(lon_step, lat_step):  # creates next best finer grid symmetric to borders
    num_lon = int(np.ceil(360 / lon_step))
    lon_border = (360 - (num_lon - 1) * lon_step) / 2
    num_lat = int(np.ceil(180 / lat_step))
    lat_border = (180 - (num_lat - 1) * lat_step) / 2
    lon = np.linspace(-180 + lon_border, 180 - lon_border, num_lon)
    lat = np.linspace(-90 + lat_border, 90 - lat_border, num_lat)
    return lon, lat


def cartesian2spherical(x, y, z):
    """Cartesian coordinates to lon and lat.

    Args:
    -----
    x: float or np.ndarray
    y: float or np.ndarray
    z: float or np.ndarray
    """
    lat = np.degrees(np.arcsin(z))
    lon = np.degrees(np.arctan2(y, x))

    return lon, lat


def spherical2cartesian(lon, lat):
    lon = lon * 2 * np.pi / 360
    lat = lat * np.pi / 180
    x = np.cos(lon) * np.cos(lat)
    y = np.sin(lon) * np.cos(lat)
    z = np.sin(lat)
    return x, y, z


def gdistance(pt1, pt2, radius=6371.009):
    lon1, lat1 = pt1[1], pt1[0]
    lon2, lat2 = pt2[1], pt2[0]
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


@np.vectorize
def haversine(lon1, lat1, lon2, lat2, radius=1):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def degree2distance_equator(grid_step, radius=6371):
    """Get distance between grid_step in km"""
    distance = haversine(0, 0, grid_step, 0, radius=radius)
    return distance


def distance2degree_equator(distance, radius=6371):
    return distance * 360 / (2 * np.pi * radius)


def neighbor_distance(lon, lat, radius=6371):
    """Distance between next-nearest neighbor points on a sphere.
    Args:
    -----
    lon: np.ndarray
        longitudes of the grid points
    lat: np.ndarray
        latitude values of the grid points

    Return:
    -------
    Array of next-nearest neighbor points
    """
    distances = []
    for i in range(len(lon)):
        d = haversine(lon[i], lat[i], lon, lat, radius)
        neighbor_d = np.sort(d)
        distances.append(neighbor_d[1:2])

    return np.array(distances)


def min_dists(grid1, grid2=None):
    if grid2 is None:
        lon1, lon2 = grid1['lon'], grid1['lon']
        lat1, lat2 = grid1['lat'], grid1['lat']
        d = 9999 * np.ones((len(lon1), len(lon2)))
        for i in range(len(lon1)):
            for j in range((len(lon1))):
                if i < j:
                    d[i, j] = gdistance((lat1[i], lon1[i]), (lat2[j], lon2[j]))
                elif i > j:
                    d[i, j] = d[j, i]
        return d.min(axis=1)
    else:  # min dist from self.grid point to other grid
        lon1, lon2 = grid1['lon'], grid2['lon']
        lat1, lat2 = grid1['lat'], grid2['lat']
        d = 9999 * np.ones((len(lon1), len(lon2)))
        for i in range(len(lon1)):
            for j in range(len(lon2)):
                d[i, j] = gdistance((lat1[i], lon1[i]), (lat2[j], lon2[j]))
        return d.min(axis=1)

cartesian_grid = spherical2cartesian(lon = np.random.randn(10),lat = np.random.randn(10))

# %%
from sklearn.gaussian_process.kernels import Matern
import time


cartesian_grid = spherical2cartesian(lon = np.random.randn(10),lat = np.random.randn(10))

num_runs = 1
nus = [0.5, 1.5, 2.5]
len_scales = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ar_coeff = 0.9
n_time = 1000
base_path = 'C:/Users/Acer/PycharmProjects/fugw/data/'
ar = int(ar_coeff*10)
grid_type = 'sph'
n_lat = 10


def diag_var_process(ar_coeff, cov, n_time):
    """Generate a time series with a given covariance matrix and AR(1) process.
    Args:
    -----
    ar_coeff: float
        AR(1) coefficient
    cov: np.ndarray
        Covariance matrix
    n_time: int
        Number of time points
    """
    n_gridpoints = cov.shape[0]
    data = np.zeros((n_time, n_gridpoints))
    data[0, :] = np.random.multivariate_normal(np.zeros(n_gridpoints), cov)
    for t in range(1, n_time):
        data[t, :] = ar_coeff * data[t - 1, :] + np.random.multivariate_normal(np.zeros(n_gridpoints), cov)
    return data


def mysave(param, param1, data):
    if not os.path.exists(param):
        os.makedirs(param)
    np.savetxt(param + param1, data, delimiter=',')
    return data


for nu in nus:
    for len_scale in len_scales:
        # generate data with chordal Matern covariance
        kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
        cov = kernel(cartesian_grid)
        for irun in range(num_runs):
            seed = int(time.time())
            np.random.seed(seed)
            data = diag_var_process(ar_coeff, cov, n_time)
            mysave(base_path+'empdata/', f'data_matern_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_var1_seed{seed}.txt',data)


# now data is a n_time x n_gridpoints matrix of the time series you generated. FUGW should recover the original distance matrix D_ij. Test that...

# %%
import numpy as np


def generate_lat_lon_grid(n_lat, n_lon):
    """
    Generate a grid of latitude and longitude values.

    Parameters
    ----------
    n_lat : int
        Number of latitude divisions.
    n_lon : int
        Number of longitude divisions.

    Returns
    -------
    latitudes, longitudes : tuple of np.ndarray
        1D arrays containing the latitude and longitude values of the points on the grid.
    """
    lat_values = np.linspace(-90, 90, n_lat)
    lon_values = np.linspace(-180, 180, n_lon)

    latitudes, longitudes = np.meshgrid(lat_values, lon_values, indexing='ij')

    return latitudes.flatten(), longitudes.flatten()


n_lat = 10  # Number of latitude divisions
n_lon = 10  # Number of longitude divisions

latitudes, longitudes = generate_lat_lon_grid(n_lat, n_lon)
print(latitudes)
print(longitudes)


def haversine_distance(lat1, lon1, lat2, lon2, R=6371):
    """
    Calculate the great-circle distance between two points on a sphere using the Haversine formula.

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : float
        Latitude and longitude of the two points in degrees.
    R : float
        Radius of the sphere (default: 6371 km, Earth's average radius).

    Returns
    -------
    float
        The great-circle distance between the two points in kilometers.
    """
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

def compute_distance_matrix(latitudes, longitudes):
    n_points = len(latitudes)
    distance_matrix = np.zeros((n_points, n_points))

    for i in range(n_points):
        for j in range(i + 1, n_points):
            distance = haversine_distance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix

# Assuming latitudes and longitudes are 1D arrays with the latitudes and longitudes of the points on the spherical grid
distance_matrix = compute_distance_matrix(latitudes, longitudes)

plt.imshow(distance_matrix)
plt.colorbar()
plt.show()