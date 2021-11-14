import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

N = norm.cdf

def BS_CALL(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r*T) * N(d2)

def BS_PUT(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma* np.sqrt(T)
    return K*np.exp(-r*T)*N(-d2) - S*N(-d1)


if __name__ == "__main__" :
    K = 100
    r = 0.1
    T = 1
    sigma = 0.3

    S = np.arange(60, 140, 0.1)

    calls = [BS_CALL(s, K, T, r, sigma) for s in S]
    puts = [BS_PUT(s, K, T, r, sigma) for s in S]
    plt.plot(S, calls, label='Call Value')
    plt.plot(S, puts, label='Put Value')
    plt.xlabel('$S_0$')
    plt.ylabel(' Value')
    plt.legend()