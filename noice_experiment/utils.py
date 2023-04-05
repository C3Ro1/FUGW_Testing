import numpy as np


class AdjGeodConverter:

    def __init__(self, adjacency=None):
        if adjacency is None:
            raise ValueError('Saruman the white? Saruman the fool!')
        self.adjacency = adjacency
        self.geodesic = self.generate_geodesic()
        pass

    def generate_geodesic(self):
        distance_matrix, ad_list, weights = self.floyd_warshall()
        return distance_matrix, ad_list, weights

    # All source shortest path algorithm: Floyd-Warshall

    def floyd_warshall(self):
        n = len(self.adjacency)
        adj_list = []
        dist = [[float('inf')] * n for i in range(n)]
        done = [[False] * n for i in range(n)]

        for i in range(n):
            for j in range(n):
                if self.adjacency[i][j] != 0:
                    dist[i][j] = self.adjacency[i][j]
                    if not done[i][j] is None:
                        adj_list.append([i, j])
                        done[i][j] = done[j][i] = True
                    pass
                    pass
                pass
            pass
        print(adj_list)
        weight_list = []

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if i == j:
                        dist[i][j] = 0
                    else:
                        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
                        dist[j][i] = dist[i][j]
                        pass
                    pass
                    if k == n - 1:
                        if i < j and self.adjacency[i][j] != 0:
                            weight_list.append(dist[i][j] if dist[i][j] != float('inf') else 0)
        return dist, adj_list, weight_list


def spherical2cartesian(lon, lat):
    lon = lon * 2 * np.pi / 360
    lat = lat * np.pi / 180
    x = np.cos(lon) * np.cos(lat)
    y = np.sin(lon) * np.cos(lat)
    z = np.sin(lat)
    return x, y, z


# imshow of distance matrix of unit sphere


def cartesian2distance(cartesian_3):
    shape,cartesian_3 = np.shape(cartesian_3)[1], cartesian_3.T
    return [[np.sqrt(np.dot(cartesian_3[i] -
                            cartesian_3[j],
                            cartesian_3[i] -
                            cartesian_3[j]))
             if i != j
             else 0
             for j in range(shape)]
            for i in range(shape)]


def spherical2distance(lon, lat):
    return np.transpose(spherical2cartesian(lon, lat)), cartesian2distance(np.array(spherical2cartesian(lon, lat)))

#TODO
def spherical2noisy_features():
    return None

#TODO
def pointwise_noise(input):
    return input

def hemisphere_noise(input):
    return input

noice_types = [pointwise_noise, hemisphere_noise]
