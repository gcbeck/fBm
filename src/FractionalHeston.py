from dataclasses import dataclass

import numpy as np
from math import ceil
from os.path import dirname, realpath, join, pardir

from Dzhaparidze import Dzhaparidze
from FractionalKernel import FractionalKernel

CACHEPATH = join(dirname(realpath(__file__)), pardir, 'dat')  # Default cache is a local 'dat' subdir

IEULER = 1/2.7182818
DAYVSCALE = np.sqrt(252)
HOURVSCALE =  DAYVSCALE * np.sqrt(6.5)  # Assuming 6.5 hour exchange days

@dataclass
class VolP():
    """ 
    :param eta: The vol-of-vol multiplier
    :param alpha: The Ornstein-Uhlenbeck reversion rate
    :param theta: The Ornstein-Uhlenbeck 'long-term' vol value
    """
    eta: float
    alpha: float
    theta: float

@dataclass
class FBmP(VolP):
    """ 
    :param H: The Hurst exponent of the volatility process, preferably < 0.3
    :param n: The multifactor order: number of multiscale mean reversions constituting the volatility
    """
    H: float
    n: int

class VolG(object):
    """ Encapsulates the volatility generation process for Hurst <= 0.5, given a price process
          random sequence that acts as the sole source of randomness. 
          This base class itself invokes a Brownian volatility (H=0.5). It is inherited to invoke an fBm. """
    
    def __init__(self, dW, volP):
        """ 
        :param dW: The NxM matrix of price process innovations; N the series length, M the number of series 
        :volP: The parameterization for a Brownian (Hurst=0.5) OU volatility
        """

        ialpha = 1 - volP.alpha
        vtheta  = np.log(volP.theta)
        N, M = dW.shape

        self.vol = -volP.eta * dW + volP.alpha * vtheta
        self.vol[0, :] += ialpha * vtheta
        for ix in range(1, N):
            self.vol[ix, :] += ialpha * self.vol[ix-1, :]
        np.exp(self.vol, out=self.vol)
        pass

    def get(self):
        return self.vol

class FBmG(VolG):
    """ The fractional Brownian volatility generation process for Hurst < 0.5, preferably < 0.3, 
          given a price process random sequence that acts as the sole source of randomness. """
    
    def __init__(self, dW, fBmP):
        """ 
        :param dW: The NxM matrix of price process innovations; N the series length, M the number of series 
        :fBmP: The parameterization for a fractional Brownian (Hurst<0.5 and preferably < 0.3) OU volatility
        """
        self.vol = self.multiscale(dW, fBmP)
        np.exp(self.vol + np.log(fBmP.theta), out=self.vol)

    @staticmethod
    def multiscale(dW, fBmP):
        """ 
        Fractionalizes (for H<0.5) a random series via the Rosenbaum method, approximating the long range 
        power law kernel as a quadrature that decomposes an fBm into a superposition of multi-scale mean reversions.  
        :param dW: The NxM matrix of random innovations; N the series length, M the number of series 
        :fBmP: The parameterization for a fractional Brownian (Hurst<0.5 and preferably < 0.3) OU volatility
        """
        vtheta  = np.log(fBmP.theta)
        N, M = dW.shape
        rng = np.random.default_rng()

        out = dW * (-fBmP.eta*vtheta)

        fracKernel = FractionalKernel(fBmP.n, fBmP.H, cachePath=CACHEPATH)
        ccoefs, gcoefs = fracKernel.getCCoefs(), fracKernel.getGCoefs()
        z = np.transpose(rng.normal(0, fBmP.eta*np.sqrt((1 - np.exp(-2*(gcoefs + fBmP.alpha*ccoefs))) 
                                                                                                    / (2 * (gcoefs + fBmP.alpha*ccoefs))), size=(M, fBmP.n)))
        istep = np.tile(1 / (1 + gcoefs[:, np.newaxis]), M)

        z = np.multiply(istep, z + np.tile(out[0, :], (fBmP.n, 1))) 
        out[0, :] = np.matmul(ccoefs, z)
        for ix in range(1, N):
            z = np.multiply(istep, z + np.tile(np.multiply(out[ix, :], 1 + out[ix-1, :]/vtheta)  
                                                                     - fBmP.alpha*out[ix-1, :], (fBmP.n, 1))) 
            out[ix, :] = np.matmul(ccoefs, z)
        
        return out


class FractionalHeston(object):
    """ Uses the Rosenbaum multifactor method (when Hurst < 1/2, or a Brownian process if H=1/2)
          to generate a fractional stochastic volatility which acts as the vol factor in another fBm price process 
          of any 0 < Hurst < 1 """
    
    def __init__(self, N, M, volP, H, n=None):
        """ 
        :param N: Length of (each) (doubly-)fBm series to generate
        :param M: The number of fBm series to generate (and hence the number of cols in the output matrix)  
        :param volP: The parameterization of the volatility, whether fBm or Brownian
        :param H: The Hurst exponent of the price process, likely > 0.5
        :param n: The number of 'infinite' series terms to use. Fewer means greater truncation of higher frequencies
                          'None' triggers use of a heuristic linking N and n
        """

        if n is None:
            n = 2 ** ceil(4 + IEULER*np.log2(N))
        dW = np.zeros((N, M))
        dW[1:, :] = np.diff(Dzhaparidze(N, M, n, H, cachePath=CACHEPATH).get(), axis=0)

        volG = FBmG(dW, volP) if hasattr(volP, 'H') else VolG(dW, volP)

        self.S = np.exp(np.cumsum(-0.5*np.square(volG.get()) + (volP.eta*volP.theta) + np.multiply(volG.get(), dW), axis=0))

    def drift(self, driftP):
        """ 
        Adds a fractional (H<0.5) random drift computed as a superposition of multiscale mean reversions
        :param driftP: The fBm parameterization of the drift
                      driftP.eta: The volatility of the drift
                      driftP.theta: The Ornstein-Uhlenbeck 'long-term' drift volatility
                      driftP.alpha: The Ornstein-Uhlenbeck drift reversion rate
        """
        dW = np.random.default_rng().normal(0, 1, size=self.S.shape)
        self.S *= np.exp(FBmG.multiscale(dW, driftP))

    def get(self, So=1):
        return self.S * So


def main():
    import matplotlib.pyplot as plt
    
    # Generate a series of Heston price processes
    
    #prices = FractionalHeston(1024, 4, VolP(eta=0.005, alpha=0.1, theta=0.08/DAYVSCALE), 0.6)
    prices = FractionalHeston(1024, 4, FBmP(eta=0.005, alpha=0.2, theta=0.07/DAYVSCALE, H=0.2, n=8), 0.6)

    # Add a random drift 
    #prices.drift(FBmP(eta=0.0001, alpha=0.04, theta=0.06/DAYVSCALE, H=0.24, n=8))

    plt.plot(prices.get(So=4096))
    plt.show()
    

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    main()

        