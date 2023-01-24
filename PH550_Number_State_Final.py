# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 19:18:19 2022

@author: sjmcg
"""

#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
# from matplotlib.ticker import MaxNLocator


#parameters
K = 10 # number of modes total
L = 1 # number of modes that are being detected (L/K is the quantum efficiency)
N = 2 # Number of photons emitted per coherence time (assumed fluctuation free)
intNo = 1 # integration time / coherence time


# I have not yet worked out an analytical expression for M
# We know however that it is less than N^K, we can determine M empirically later

states = np.zeros((N**K,K))

# "states" contains the number states explicitly
# storing them all is relatively wasteful with memory, but it is nicely illustrative, so we do it for now

# Now we have to fill the photon numbers into the states
# we use a recursive function for this
m = 0



# this is our recursive function
# It recursively "propagates" the number states from left to right
def fill_states(current_state, k, K, n):
    global states
    global m
    if k==K-1: # we are at the last mode, all remaining photons must go into it
        current_state[0,k] = n
        states[m,:] = current_state
        m = m+1
        return
    elif n==0: # all photons are already distributed, we can stop here
        current_state[0,k:K] = 0
        states[m,:] = current_state
        m = m+1
        return
    else:
        n_k = n
        while n_k >= 0: # n_k goes through all possibilities
            current_state[0,k] = n_k
            fill_states(current_state, k+1, K, n-n_k)
            n_k = n_k-1

init_state = np.zeros((1,K))
fill_states(init_state, 0, K, N)
M = m # M is determined empirically here
# let us print it out what M is
print(M)

states = states[0:M,:]

#%% Steve code starting here
# This code below will select a number ofstates corresponding to QE of L/K
df = pd.DataFrame(states)
chosenStates = df.iloc[:, : (int((L/K)*10))]
MsumOfSCS = np.zeros(len(chosenStates)*10)
i=0
while i<10:
    shuffleChosenStates = chosenStates.sample(frac=1).reset_index(drop=True) #reset index removes old index values
    sumOfSCS = shuffleChosenStates.sum(axis=1)
    MsumOfSCS[i*len(chosenStates):(i+1)*len(chosenStates)] = sumOfSCS
    i=i+1
MsumOfSCS = pd.DataFrame(MsumOfSCS)
# Sum together by integration number ()
intSums = MsumOfSCS.groupby(MsumOfSCS.index // intNo).sum()


# randomState = (pd.Series([np.random.choice(i,1)[0] for i in df.values]))
# randomStateFull = randomState


# This code will select a random number of the rows to return, removing 20% at random
# remove_n = int(0.2*len(randomState))
# drop_indices = np.random.choice(randomState.index, remove_n, replace=False)
# randomState = randomState.drop(drop_indices)
# randomState = np.array(randomState)


# # Random index for state selection
# index = int(np.floor(K*np.random.rand(1)))

# # Use index to choose one of the propagated states
# randomState = states[:,index]

# For small numbers of N and K you can display the number states by uncommenting the next line
if N<5 and K<5:
    print(states)

# now, for each state we can calculate the detected photons

D = np.sum(states[:,0:L], axis =1)


# again, if we have sufficiently few states, we can have a look at the values
if N<5 and K<5:
    print(D)


# fill the probabilities
p = np.ones((1,M))/M

# calculate expectation value
Dexp = np.sum(p*D)
print(Dexp)


#%% Histograms of Random States %%#
# next we build a histogram
# this part will need to be revised if the probabilities p_i are not all the same
# plt.figure()


intSumsArray = np.array(intSums)
intSumsArraySD = np.std(intSums)
intSumsX = np.arange(1,len(intSumsArray)+1,1)
intSumsArray = np.reshape(intSumsArray,len(intSumsX))
plt.style.use('bmh')


# plot bar
ax = plt.figure().gca()
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.rcParams["figure.figsize"] = (12,8)
plt.title('Photon Counts by Integration Event of Random Selection of ' + str(intNo))
plt.bar(intSumsX, intSumsArray, align='center', alpha=0.5)
plt.ylabel("Number of counted photons", fontsize=28, fontweight='bold')
plt.xlabel("Integration Event", fontsize=28, fontweight='bold')
plt.tick_params(axis='both', labelsize=20)
plt.show()
# plt.savefig('/Users/stevenconnan-mcginty/Pictures/Fig1.png')



# # Plot histogram
# plt.figure()

hist,hist_bins = np.histogram(intSumsArray, bins=np.arange(N*intNo+2))
hist = hist/((M/intNo)*10) # normalise so that the values represent probabilities
hist = hist.round(decimals=3)
hist_n = np.arange(N*intNo+1) # will be the y-axis for the plot, corresponds to number of photons detected
H_mean = np.mean(intSumsArray)
H_SDlow = H_mean - np.std(intSumsArray)
H_SDhigh = H_mean + np.std(intSumsArray)


# plot histogram
ax = plt.figure().gca()
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.rcParams["figure.figsize"] = (12,8)
# plt.title('Histogram of all states')
plt.bar(hist_n, hist, align='center', alpha=1)
plt.xlabel("Number of detected photons (D)", fontsize=28, fontweight='bold')
plt.ylabel("Probability of D photons", fontsize=28, fontweight='bold')
plt.tick_params(axis='both', labelsize=20)
plt.show()
plt.axvspan(H_SDlow, H_SDhigh, color='green', alpha=0.25, lw=0)
plt.axvline(H_mean, linestyle='dashdot', color='black')
intSumsArraySD = np.std(intSums)
# plt.savefig('/Users/stevenconnan-mcginty/Pictures/Full System - .png')

#%%
# plot bar
ax = plt.figure().gca()
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.rcParams["figure.figsize"] = (12,8)
# plt.title('Photon Counts by Integration Event of Random Selection of ' + str(intNo))
plt.scatter(intSumsX, intSumsArray, s=150, alpha=1)
plt.ylabel("Number of counted photons", fontsize=28, fontweight='bold')
plt.xlabel("Integration Event", fontsize=28, fontweight='bold')
plt.tick_params(axis='both', labelsize=20)
plt.axhline(H_mean, 0, len(intSumsArray), linestyle='dashdot', color='black')
plt.show()
# plt.savefig('/Users/stevenconnan-mcginty/Pictures/Integrations - .png')

