import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


class StochasticBlock:
    def __init__(self, number_of_nodes=None, number_of_blocks=None, p=None, seed=None):
        self.nodes = 3 if number_of_nodes is None else number_of_nodes
        self.blocks = 4 if number_of_blocks is None else number_of_blocks
        self.p = [0.8] if p is None else p
        self.G = nx.Graph()
        self.model = None
        self.seed = 10 if seed is None else seed
        pass

    def plot(self, model_mode=True):
        if not model_mode:
            nx.draw(self.G, with_labels=True)
            plt.show()



        pass

    def generate_model(self):
        '''
        Generates the stochastic block model used to sample
        the graph.
        returns the adjacency matrix

        Parameters:
        ----------
        None

        Returns:
        ----------
        A: np.array     | The adjacency matrix of the graph
        '''
        models = []

        for p in self.p:
            q = 1 - p
            self.G = nx.stochastic_block_model(
                [self.nodes for _ in range(self.blocks)],
                [[p if i == j else q for i in range(self.blocks)] for j in range(self.blocks)]
                , seed=self.seed)
            models.append(nx.to_numpy_array(self.G))
        self.model = models
        pass

    def generate_graph(self):
        '''
        Generates the graph from the model

        Parameters:
        ----------
        model: np.array     | The adjacency matrix of the stochastic block model

        Returns:
        ----------
        G: nx.Graph         | The randomised graph containing geodesic distances
                            | for each vertex
        '''

        return self.G

    def permutation_sort(self):
        '''
        Sorts the model such that the stochastic block model is insensitive to
        permutation of the nodes.

        Therefore the scalar product of each column is calculated from one
        point such that the columns are sorted in ascending order.

        Parameters:
        ----------
        model: np.array     | The adjacency matrix of the stochastic block model

        Returns:
        ----------
        model: np.array     | The sorted adjacency matrix of the stochastic block model
        '''

        scalars = merge_sort_2d(
            [[np.vdot(self.model[0][0], b) for b in self.model[0][:]],
             [i for i in range(len(self.model[0][:]))]])
        edges = []
        d_permute = dict()
        d_reverse = dict()

        for j, i in enumerate(scalars[1]):
            temp = []
            d_permute[j] = i
            d_reverse[i] = j
            for k in range(self.nodes * self.blocks):
                if self.model[0][i][k] == 1:
                    temp.append(k)
            edges.append(temp)
            pass

        scalars = [d_permute,d_reverse, edges]
        self.realign_model(scalars)

        pass

    def realign_model(self, scalars):
        m = self.model[0]
        plt.imshow(m)
        plt.colorbar()
        plt.show()

        for i in scalars[1].values():
            for j in scalars[2][i]:
                self.model[0][i][j] = m[scalars[0][i]][scalars[1][j]]
                pass
            pass
        pass

        m = self.model[0]
        plt.imshow(m)
        plt.colorbar()
        plt.show()


def merge_sort_2d(array):
    if len(array[0]) <= 1:
        return array
    mid = len(array[0]) // 2
    left = [array[0][:mid], array[1][:mid]]
    right = [array[0][mid:], array[1][mid:]]
    left = merge_sort_2d(left)
    right = merge_sort_2d(right)
    return merge_2d(left, right)


def merge_2d(left, right):
    result = [[], []]
    i, j = 0, 0
    while i < len(left[0]) and j < len(right[0]):
        if left[0][i] >= right[0][j]:
            result[0].append(left[0][i])
            result[1].append(left[1][i])
            i += 1
        else:
            result[0].append(right[0][j])
            result[1].append(right[1][j])
            j += 1
    result[0] += left[0][i:]
    result[1] += left[1][i:]
    result[0] += right[0][j:]
    result[1] += right[1][j:]
    return result


if __name__ == "__main__":
    model = StochasticBlock()
    model.generate_model()
    model.permutation_sort()
    pass
