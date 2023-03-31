from matplotlib.pyplot import plot,draw,legend,show
from numpy import linspace


class experiment_one_noise():

    def plot_for_alphas(self, out_loss, alpha_levels, noise_resolution, repetitions_of_noise = None):

        if repetitions_of_noise is None:
            #7*0:12 + 1
            #7*0 -> 7*1 -> 7*2 -> ... -> 7*12
            #7*13 ->

            for current_alpha in range(alpha_levels):
                plot_out = []
                for level_of_noise in range(noise_resolution):
                    plot_out.append(out_loss[alpha_levels*level_of_noise+current_alpha])
                    pass
                plot(linspace(0.5,1.0,noise_resolution),plot_out)
                pass
            show()
        else:
            for repetition in range(repetitions_of_noise):
                for current_alpha in range(alpha_levels):
                    plot_out = []
                    for level_of_noise in range(noise_resolution):
                        plot_out.append(out_loss[alpha_levels*
                                                 ((noise_resolution*repetition)+level_of_noise)+current_alpha])
                        pass
                    plot(linspace(0.5,1.0,noise_resolution),plot_out)
                    pass
            show()


