class GeometricNoise:
    '''
    This class is used to test the FUGW framework
    with geometric noise. It is the first experiment
    in the extensive testing section of the thesis.

    Geometric noise stability is tested by applying
    a stochastic block model to a graph and then
    adding geometric noise to the graph. The
    stability of the FUGW framework is then tested
    by comparing the results of the FUGW framework
    with and without geometric noise.

    The Results of this experiment are presented in
    form of heatmaps. The heatmaps show the optimal
    pairing produced by the FUGW framework. The data
    is also presented as plotted graphs, before and
    after the FUGW framework is applied.

    In a similar fashion, the results of the FUGW, are
    presented in a plotted graph, where the x-axis shows
    the FUGW-loss while increasing the noise level.

    Besides the stochastic block model, the experiment
    also uses the Erdos-Renyi model to test the FUGW
    stability.



    Parameters:
    ----------
    model: np.array     | The adjacency matrix of the stochastic block model

    Returns:
    ----------
    None

    '''
    def __init__(self, noise_model = None):
        self.noise_model = noise_model


        pass