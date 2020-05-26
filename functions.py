import random as rand
import numpy as np
import scipy.integrate as integrate 
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.optimize import curve_fit
from tqdm import tqdm  

def convolve(dist1,dist2):
    new = np.zeros_like(dist1)
    
    for n in range(len(new)):
        val = 0
        for m in range(n):
            val = val + dist1[m]*dist2[n-1-m]      
            new[n] = val

    return new

def singleAtom(recordTime,rate,lifeTime,xx):
    """
    analytically evaluates the probability distribution of photon counts n (xx[n]) for a single atom in a dipole trap
    Args:
        recordTime: the time that the measurment beams are active(seconds).
        rate: the rate that the single atom emits photons(photons/second)
        lifeTime: the lifetime of the trap(seconds)
        xx: an array of integers that are the possible photon counts to measure
    Returns:
        an array of probabilities to measure the photon count given the paramaters.

    """ 
    prob = [] 
    #go through all of the photon counts in xx
    for i in tqdm(xx):
        #take the integral of the lifetime and the poisson dist 
        val = integrate.quad(
            lambda t,r,n,tau:np.exp(-t/tau)/tau*poisson.pmf(n,r*t),
            0,
            recordTime,
            args = (rate,i,lifeTime) 
        )[0] 
        prob.append(val+np.exp(-recordTime/lifeTime)*poisson.pmf(i,rate*recordTime)) #add in the 

    
    return prob


def getHistogram(probDistBack,probDistSingle,photonArray,dataPoints = 5000,samples = [], weights = [.5,.5]):
    """
    calcuates an array of counts given the background and atom distributions
    Args:
        probDistBack: the background probability distribution
        probDistSingle: the single atom probability distribution
        dataPoints: the amount of times to run the experiemnt
        samples: an array of length boolean values that represent if that atom is trapped or not
        weights: the loading proability for 0,1 atoms
    Returns:
        an array of values that are ready for a histogram

    """ 
    hist = [] 
    population = [False,True] #false is no atom, true is single atome
    #check to see if we are supplied with any samples, if not generate with dataPoiints
    if len(samples) == 0:   
        samples = rand.choices(population,weights,k = dataPoints)

    for i in samples:
        if not i:
            #do background rate
            hist.append(rand.choices(photonArray,probDistBack)[0])
        else:
            #do single atom rate
            hist.append(rand.choices(photonArray,probDistSingle)[0])
    return hist

#the function for scipy curve fit as the addition of two gaussian curves
def gaussian2(x, amp1,cen1,sigma1,amp2,cen2,sigma2):
    """ 
    a function of the addition of two gaussians
    Args:
        x: position to evaluate the function at
        amp1: the amplitude of the first gaussian function
        cen1: the center of the first gaussian function
        sigma1: the standard deviation of the first gaussian function
        amp2: the amplitude of the second gaussian function
        cen2: the center of the second gaussian function
        sigma2: the standard deviation of the second gaussian function
    Returns:
        the value of the double gaussian at position x

    """
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2))) + \
            amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen2)/sigma2)**2)))
def gaussian1(x, amp1,cen1,sigma1):
    """ 
    a function of the addition of one gaussians
    Args:
        x: position to evaluate the function at
        amp1: the amplitude of the first gaussian function
        cen1: the center of the first gaussian function
        sigma1: the standard deviation of the first gaussian function
       
    Returns:
        the value of the single gaussian at position x

    """
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2)))
            
def graph(histogram,photonArray, p0 = [],binNumber = 50, colors = ["gray", "red"],fit = True,double = True):
    """ 
    Args:
        histogram: the dist of data to put in the histogram
    Kwargs:  
        p0: a list of the initial guesses of for the scipy curve fit function on two gaussian functions 
            [amplitude1,center1,sigma1,amplitude2,center2,sigma2]     
        binNumnber: the number of bins in the histogram
        colors: a list of the colors for the histogram chart and fit line
        fit: a boolean that when true graphs a fit from the p0 values
        Double: a boolean that when true fits a double gaussian to the data
    Returns:
        returns the y values of the fit for the given range of the histogram data, if no fit is given then an 
        empty array is returned
    """
    #the histogram and the bin heights and centers that will be used for fitting
    bin_heights, bin_borders, _ = plt.hist(histogram,binNumber,color = colors[0])

    if fit:
        bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
        #fit with guess p0
       
        x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
        if double:
            popt, pcov = curve_fit(gaussian2, bin_centers, bin_heights, p0)
            plt.plot(x_interval_for_fit, gaussian2(x_interval_for_fit, *popt), 
                label=r"$\mu_1$ = %.2f std = %.2f $\mu_1$ = %.2f std = %.2f"%(
                    popt[1],popt[2],popt[4],popt[5]),color = colors[1]
                    )
        else:
            popt, pcov = curve_fit(gaussian1, bin_centers, bin_heights, p0)
            plt.plot(x_interval_for_fit, gaussian1(x_interval_for_fit, *popt), label=r"$\mu_1$ = %.2f and std = %.2f"%(popt[1],popt[2]),color = colors[1])
        plt.legend()    
    
    plt.xlabel("Photon Counts")
    plt.ylabel("Frequency of Counts")
    plt.show()
    if not fit:
        return []
    if not double:
        return gaussian1(photonArray,*popt)
    else:
        return gaussian2(photonArray,*popt)
