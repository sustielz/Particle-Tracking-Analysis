from matplotlib import pyplot as plt
import numpy as np
# import pandas as pd
from scipy.optimize import leastsq
from scipy.optimize import curve_fit


#### Methods for analyzing Particle Oscillation 
def fourier(x, FPS=30):    #### Fourier Transform. Assume no 'missing' frames.
    num_frames = np.size(x)
    fx = np.fft.rfft(x)*np.conjugate(np.fft.rfft(x))
    fx[0] = 0.0   #### Drop zero mode
    hz = FPS/num_frames*np.arange(np.size(fx))
    return hz, fx


def sinfunc(n, params):
    amp=params[0]
    freq=params[1] 
    phase=params[2]
    return amp*np.sin(2*np.pi*freq*n + phase)


def modsinfunc(n, params):
    amp=params[0]
    freq=params[1] 
    phase=params[2]
    depth = params[3]
    rate = params[4]
    shift = params[5]
    S = sinfunc(n, [amp, freq, phase])
    M = sinfunc(n, [1, rate, shift])
    return S*(1 - depth/2 + (depth/2)*M)


def sinfit(x, MOD=False, k0=0, framenum=None, guess=None, FPS=30): ## Fit to a sin (or amp-modulated sin) function. 
    guess = [max(x), k0, 0., 0.5, 1, 0.] if guess is None else guess
    framenum = np.arange(np.size(x)) if framenum is None else framenum
    t = framenum/FPS
    
    func = modsinfunc if MOD else sinfunc
    def residual(params, data, n):
        return data - func(n, params)
    
    fitvals, trash = leastsq(residual, guess, args=(x, t))
#     return X, trash
    return fitvals, func(t, fitvals)  


    
def expfunc(n, params):
    A = params[0]
    tau = params[1]
    return A*np.exp(-n/tau)
    
def expfit(C, framenum=None, guess=None, FPS=30):
    guess = [C[0], 2/FPS] if guess is None else guess
    framenum = np.arange(np.size(C)) if framenum is None else framenum
    t = framenum/FPS    
        
    func = expfunc
    def residual(params, data, n):
        return data - func(n, params)
    
    fitvals, trash = leastsq(residual, guess, args=(C, t))
    #   return X, trash
    return fitvals, expfunc(t, fitvals)   

#### Methods for analyzing Correlation Functions
def corr(x1, x2):
    N = np.size(x1)
    out = np.zeros(N)
    for i in range(N):
        for j in range(N-i):
            out[i] += x1[j]*x2[j+i]
        #out[i] = out[i]/(N-i)
    out = out/(N-np.arange(N))
    return out 
#     return out/out[0]

def acorr(x, n=None):    
    return corr(x, x) if n is None else corr(x, x)[:n]



def phaseplot(x, FL, f=6.14, dt=20/1000, FPS=30, label=' '):
    phi0 = np.array(FL[0])/(2*np.pi*f)
    t = np.arange(np.size(x))/FPS
    
    params, X = sinfit(x, k0=f)
#     plt.plot(t, x)
#     plt.plot(t, X, linestyle=':')
    color=next(plt.gca()._get_lines.prop_cycler)['color']
    plt.plot(t, x, color=color)
    plt.plot(t, X, linestyle=':', color=color)
    for ivl in FL:
        plt.axvline(x=ivl[0], color='green')
        plt.axvline(x=ivl[1], color='green')

    x0 = params[3]/(2*np.pi*f)
    while x0 < t[-1]:
        plt.axvline(x=x0, color='black')
        x0 += 1/f        
    str1 = '{}: f={:.3f} | FIT: A={:.3f}, f={:.3f}, phi={:.3f}'.format(label, f, params[0], params[1], params[2])
    str2 = ' -> dphi ~ [{}, {}]'.format(params[2] - phi0[0], params[2] - phi0[1])
    print(str1 + str2)
    
    return params, phi0
