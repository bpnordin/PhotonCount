import numpy as np
import matplotlib.pyplot as plt
from random import choices
#the probability of leaving a trap is an exponential decay, so we will define the function here
def decay(x,amp,center,rate):
    return amp*np.exp((x-center)*rate)

def generateHistogram(amp,center,rate,timeStop = .03,length = 800,numberOfChoices = 1000,bins = 50,
                        name = "randomleaving"):
    xx = np.linspace(0,timeStop,length)
    probArray = decay(xx,amp,center,rate)
    norm = np.linalg.norm(probArray)
    probArray = probArray*norm
    nPhotons = 0
    array = []
    backgroundRate = 10
    singleRate = 1000
    #get a bunch of samples
    choice_array = choices(xx,probArray,k = numberOfChoices)
    for c in choice_array:
        #we will leave after c seconds
        nPhotons = 0

        for t in xx:
            if t >= c:
                #we have left the trap
                nPhotons += backgroundRate
            else:
                nPhotons += singleRate    
        array.append(nPhotons/length)
    plt.hist(array ,bins)
    plt.savefig(name)

generateHistogram(10,0,-100)
