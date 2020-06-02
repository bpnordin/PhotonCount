import random as rand
import numpy as np
import scipy.integrate as integrate 
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.optimize import curve_fit
from tqdm import tqdm  

def convolve(dist1,dist2):
    """
    convolves two probability distributions of the same size 
    Paramaters
    ----------
    dist1 : array_like
        an array to be convolved with the other array in the argument
    dist2 : array_like
        an array to be convolved with the other array in the argument
    """
    new = np.zeros_like(dist1)
    
    for n in range(len(new)):
        val = 0
        for m in range(n):
            val = val + dist1[m]*dist2[n-1-m]      
            new[n] = val

    return new

def singleAtom(recordTime,rate,lifeTime,photonArray):
    """
    Analytically evaluates the probability distribution of photon counts n (photonArray[n]) for a single atom in a dipole trap

    Paramaters
    ----------
    recordTime : int 
        the time that the measurment beams are active(seconds).
    rate : int 
        the rate that the single atom emits photons(photons/second)
    lifeTime : int 
        the lifetime of the trap(seconds)
    photonArray : array_like
        an array of integers that are the possible photon counts to measure

    Returns
    -------
        an array of probabilities to measure the photon count given the paramaters.
    """ 

    prob = [] 
    #go through all of the photon counts in photonArray
    for i in tqdm(photonArray):
        #take the integral of the lifetime and the poisson dist 
        val = integrate.quad(
            lambda t,r,n,tau:np.exp(-t/tau)/tau*poisson.pmf(n,r*t),
            0,
            recordTime,
            args = (rate,i,lifeTime) 
        )[0] 
        prob.append(val+np.exp(-recordTime/lifeTime)*poisson.pmf(i,rate*recordTime)) #add in the 

    
    return prob


def runExperiment(probDistBack,probDistSingle,photonArray,dataPoints = 5000,samples = None, weights = [.5,.5]):
    """
    Calcuates an array of counts given the background and atom distributions
    
    Paramaters
    ----------
        probDistBack : array_like 
            the background probability distribution
        probDistSingle : array_like 
            the single atom probability distribution
        dataPoints: int
            the amount of times to run the experiemnt
        samples: array_like
            an array of length boolean values that represent if that atom is trapped or not
            this will autogenerate if left blank
        weights: length 2 array
            the loading proability for 0,1 atoms
    Returns:
        an array of values that are ready for a histogram

    """ 
    population = [False,True] #false is no atom, true is single atome
    #check to see if we are supplied with any samples, if not generate with dataPoiints
    if samples is None:
        samples = rand.choices(population,weights,k = dataPoints)
    background = rand.choices(photonArray,probDistBack,k = dataPoints)
    singleAtom = rand.choices(photonArray,probDistSingle,k=dataPoints)

    hist = [background[i] if not loaded else singleAtom[i] for i,loaded in enumerate(samples)]

    return hist

#the function for scipy curve fit as the addition of two gaussian curves
def gaussian2(x, amp1,cen1,sigma1,amp2,cen2,sigma2):
    """
    the addition of two gaussians
    Paramaters
    --------- 
    a function of the addition of two gaussians
    Args:
        x : int
            position to evaluate the function at
        amp1 : int
            the amplitude of the first gaussian function
        cen1 : int
            the center of the first gaussian function
        sigma1 : int
            the standard deviation of the first gaussian function
        amp2 : int
            the amplitude of the first gaussian function
        cen2 : int
            the center of the first gaussian function
        sigma2 : int
            the standard deviation of the first gaussian function
    Returns
    --------
    the value of the double gaussian at position x

    """
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2))) + \
            amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen2)/sigma2)**2)))

def gaussian1(x, amp1,cen1,sigma1):
    """ 
    One single gauusian
    Paramaters
    --------- 
    a function of a single gaussian
    Args:
        x : int
            position to evaluate the function at
        amp1 : int
            the amplitude of the gaussian function
        cen1 : int
            the center of the gaussian function
        sigma1 : int
            the standard deviation of the gaussian function
    Returns
    --------
        the value of the single gaussian at position x

    """
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2)))



def graphHistogram(histogram,photonArray, p0 = None,binNumber = 50, colors = ["gray", "red"],fit = False):
    """ 
    Graphs a histogram and fit of the histogram

    Paramaters
    -----------
    histogram : array_like
        the dist of data to put in the histogram
    p0 : array_like, optional
        Initial guess for the parameters (length N).  If None, then the
        initial values will all be 1 (if the number of parameters for the
        function can be determined using introspection, otherwise a
        ValueError is raised).
    binNumnber : int 
        the number of bins in the histogram
    colors : array_like
        a list of the colors for the histogram chart and fit line
    fit : boolean
        when true graphs a double gaussian fit from the p0 values 


    Returns : array
        returns the y values of the fit for the given range of the histogram data, if no fit is given then an 
        empty array is returned
    """
    #the histogram and the bin heights and centers that will be used for fitting
    bin_heights, bin_borders, _ = plt.hist(histogram,binNumber,color = colors[0])
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
    if fit:
        
        x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)

        #fit with guess p0
        popt, pcov = curve_fit(gaussian2, bin_centers, bin_heights, p0)

        plt.plot(x_interval_for_fit, gaussian2(x_interval_for_fit, *popt), 
            label=r"$\mu_1$ = %.2f std = %.2f $\mu_1$ = %.2f std = %.2f"%(
                popt[1],popt[2],popt[4],popt[5]),color = colors[1]
                )
        plt.legend()    

    plt.xlabel("Photon Counts")
    plt.ylabel("Frequency of Counts")
    plt.show()
    #return what the histogram returns
    return bin_borders,bin_heights
