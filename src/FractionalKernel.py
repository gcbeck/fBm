import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
from os import path
from warnings import warn
from scipy.special import gamma
from math import log2



class FractionalKernel(object):
    """ Uses the method in the appendix of 'Deep Calibration of the Quadratic Rough Heston Model'
          (Rosenbaum & Zhang, 2022) to find the optimal coefficient pairs of a 'rough' fBm (Hurst < 0.5) 
          approximated using multifactor  OU processes. 
          NOTE: This method is only applicable to processes with low Hurst exponents, and in fact becomes 
          more inaccurate for H above 0.3. This may cast doubt on the usefulness of rough volatility, when it
          has been shown (eg. 'Rough volatility: Fact or Artefact (Cont R & Das P, 2023) that the rough indication
          may be solely from estimation effects, and a Brownian-driven vol may work just as well. """
    
    cachePath = path.dirname(path.realpath(__file__))  # Default cache path is the file's directory 
    cacheSuffix = '.fkcg'
    hPrecision = 3  # Number of decimal places in alpha differentiated for caching
    cCoefIdx, gCoefIdx = 0, 1

    def __init__(self, n, H, N=64, M=32, X=32, T=0.1, viz=False, cachePath=None):
        """ 
        :param n: Number of terms to use in the Laplace approximation (& making up the vol term structure). 
                          Higher gives a more accurate kernel approximation (and a lower optimal 'x') but is less efficient.
                          Untested for odd integers. Typically (and minimally) something like 8. 
        :param H: The Hurst exponent. Preferably < 0.3. 
        :param N: The number of samples for candidate optimal 'x' values, ordered logarithmically between 1 and X 
        :param M: The length of discretized time vector for minimizing the mismatch between true and approximate kernels
        :param X: Upper bound for sampled x values
        :param T: Horizon for fractional kernel, smaller values reflecting interest in shorter maturity options
        """
        
        # Check if there already exists the approximate-kernel coefficient file
        hPrefix = str(round(H * (10 ** (FractionalKernel.hPrecision))))
        cache = ((cachePath if cachePath is not None else FractionalKernel.cachePath)
                        + hPrefix + '.' + str(n)  + FractionalKernel.cacheSuffix)
        if path.exists(cache):
            self.coef = np.reshape(pq.read_table(cache).to_pandas(), (n,2))
        else:
            # Minimize the reconstruction error for the power law kernel to yield an optimal 'x', which 
            # then dictates the optimal c- and (gamma) g-coefficients

            alpha = H + 0.5   # Power law exponent
            gc = gamma(alpha)
            tm = np.flip(np.linspace(T, 0, M, endpoint=False))
            K = np.tile(np.power(tm, alpha-1) / gc, (N, 1))

            gc = gc * gamma(1-alpha) * (1-alpha)
            xn = np.flip(np.logspace(log2(X), 0, N, base=2, endpoint=False))
            ci = np.multiply(np.tile((1 - np.power(xn, alpha-1)) / gc, (n, 1)), 
                                        np.power(xn, (1-alpha)*np.atleast_2d(np.arange(1-n/2, n/2+1)).transpose()))
            gi = np.multiply(np.tile(np.divide(((1-alpha) / (2-alpha)) * (np.power(xn, 2-alpha) - 1), np.power(xn, 1-alpha) - 1), (n, 1)),
                                        np.power(xn, np.atleast_2d(np.arange(-n/2, n/2)).transpose()))

            Kn = np.zeros((N, M))
            for ix in range(M):
                np.sum(np.multiply(ci, np.exp(-tm[ix] * gi)), axis=0, out=Kn[:, ix])

            xv = np.sum(np.square(Kn - K), axis=1)
            xidx = np.argmin(xv)
            xstar = xn[np.argmin(xv)]
            if np.isclose(xstar,  X):
                warn("Maximum (boundary) x value hit as argmin of objective function. Consider increasing parameter X")

            self.coef = np.stack((ci[:, xidx], gi[:, xidx]), axis=1)  # In order cCoefIdx, gCoefIdx
            pq.write_table(pa.table({"data": np.ravel(self.coef)}), cache)

            #--- Visualizations of the results -------------------------------------------------------------------------------------------
            if viz:
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(3)
                axs[0].set_title('Fractional Kernel K (True) vs Kn (Approx)') 
                axs[0].plot(tm, K[xidx, :], color='g', label='K')
                axs[0].plot(tm, Kn[xidx, :], color='b', label='Kn')
                axs[0].legend()
                axs[1].set_title('Objective Function to Minimize for Optimal x') 
                axs[1].plot(xn, xv)

                coefw = 0.3
                axs[2].bar(np.arange(n), np.log(ci[:, xidx]) , coefw, label=r"log($c_{i}^{" + str(n) + "})$")
                axs[2].bar(np.arange(n) + coefw , np.log(gi[:, xidx]), coefw, label=r"log($\gamma_{i}^{" + str(n) + "})$")
                axs[2].set_title('Log-Coefficients for Fractional Kernel Approx') 
                axs[2].set_xticks(np.arange(n) + coefw / 2, np.arange(n)+1)
                axs[2].legend()

                plt.show()
            #--- End Visualizations of the results ---------------------------------------------------------------------------------------

    def getCCoefs(self):
        return self.coef[:, FractionalKernel.cCoefIdx]

    def getGCoefs(self):
        return self.coef[:, FractionalKernel.gCoefIdx]


def main():
    import matplotlib.pyplot as plt
    n = 8
    H = 0.3

    fracKernel = FractionalKernel(n, H,)
    coefw = 0.3
    print('Log C Coefs: ')
    print(np.log(fracKernel.getCCoefs()))
    print('Log Gamma Coefs: ')
    print(np.log(fracKernel.getGCoefs()))
    plt.bar(np.arange(n), np.log(fracKernel.getCCoefs()) , coefw, label=r"log($c_{i}^{" + str(n) + "})$")
    plt.bar(np.arange(n) + coefw , np.log(fracKernel.getGCoefs()), coefw, label=r"log($\gamma_{i}^{" + str(n) + "})$")
    plt.title('Log-Coefficients for Fractional Kernel Approx') 
    plt.xticks(np.arange(n) + coefw / 2, np.arange(n)+1)
    plt.legend()


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    main()