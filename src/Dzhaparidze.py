import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
from os import path
from mpmath import besseljzero, besselj
from scipy.special import gamma

class Dzhaparidze(object):
    """ Uses the Dzhaparidze & van Zanten method '2' to give multiple fBms of Hurst H """

    cachePath = path.dirname(path.realpath(__file__))  # Default cache path is the file's directory 
    cacheNSuffix = '.fbmn'
    cacheTSuffix = '.fbmt'
    hPrecision = 3  # Number of decimal places in H differentiated for caching

    def __init__(self, N, M, T, H, cachePath=None):
        """ 
        :param N: Length of (each) fBm series to generate
        :param M: The number of fBm series to generate (and hence the number of cols in the output matrix) 
        :param T: The number of 'infinite' series terms to use. Fewer means greater truncation of higher frequencies
        :param H: The Hurst exponent
        """

        t = np.linspace(0, 1, N)
        rng = np.random.default_rng()

        # Check if there already exists the truncated series terms and/or the time/coefficient matrix
        hPrefix = str(round(H * (10 ** (Dzhaparidze.hPrecision))))
        cacheT = ((cachePath if cachePath is not None else Dzhaparidze.cachePath)
                         + hPrefix + '.' + str(T)  + Dzhaparidze.cacheTSuffix)
        if path.exists(cacheT):
            mcache = np.reshape(pq.read_table(cacheT).to_pandas(), (T,2))
        else:
            mcache = np.zeros((T, 2))    # Caches omega and sigma values 
            for ix in range(T):
                omegax = besseljzero(1-H, ix+1)
                sigmax = gamma(1-H) * np.power(omegax/2, H) * besselj(-H, omegax) 
                sigmax = sigmax * sigmax * (1-H) * gamma(1.5-H)
                sigmax = np.sqrt(H * gamma(H+0.5) * gamma(3-2*H) / sigmax)
                mcache[ix, :] = (omegax, sigmax / omegax)

            pq.write_table(pa.table({"data": np.ravel(mcache)}), cacheT)

        cacheN = ((cachePath if cachePath is not None else Dzhaparidze.cachePath)
                          + hPrefix + '.' + str(N) + '.' + str(T)  + Dzhaparidze.cacheNSuffix)
        if path.exists(cacheN):
            ncache = np.reshape(pq.read_table(cacheN).to_pandas(), (N, T*2))
        else:
            ncache = np.zeros((N, T*2))    # Caches series coefficients for both Yn and Zn
            ncache[:, 0:T] = np.outer(t, 2*mcache[:, 0])
            ncache[:, T:2*T] = np.cos(ncache[:, 0:T]) - 1
            ncache[:, 0:T] = np.sin(ncache[:, 0:T])
        
            pq.write_table(pa.table({"data": np.ravel(ncache)}), cacheN)

        # Generate the standard normal random variables X, Yn, Zn
        Yn = rng.normal(size=(T, M))
        Zn = rng.normal(size=(T, M))
        X = rng.normal(size=M)

        # Construct the fractional Brownian motion series
        self.fBm = np.matmul(ncache[:, 0:T], np.multiply(Yn, np.tile(mcache[:, 1][:, np.newaxis], (1, M))))
        self.fBm += np.matmul(ncache[:, T:2*T], np.multiply(Zn, np.tile(mcache[:, 1][:, np.newaxis], (1, M))))
        self.fBm += np.outer(t, X / np.sqrt(2 - 2*H))

        # Scale the time axis to compare to Bm: cumsum(rng.normal(size=T))
        # This is probably desired but be aware
        self.fBm *= np.power(N, H)

    def get(self):
        return self.fBm


def main():
    import matplotlib.pyplot as plt
    N = 256

    fBm = Dzhaparidze(N, 4, 128, 0.40)
    t = range(N)

    plt.plot(t, fBm.get()[:, 0], 'b')
    plt.plot(t, fBm.get()[:, 1], 'r')
    plt.plot(t, fBm.get()[:, 2], 'k')
    plt.plot(t, np.cumsum(np.random.default_rng().normal(size=N)), 'g')
    plt.show()


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    main()