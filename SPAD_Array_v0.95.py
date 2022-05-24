#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 23:28:46 2021

@author: oracle
"""
## Clear variables/objects whilst testing
#from IPython import get_ipython;
#get_ipython().magic('reset -sf')

# Ignore FutureWarnings whilst testing
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import packages for numerical manipulation and plotting
import numpy as np
import matplotlib.pyplot as plt
import math
# Import pandas to allow storing of histograms neatly
import pandas as pd 


import seaborn as sns
from scipy.stats import poisson
from scipy.stats import norm
from scipy import interpolate



# For regression
# import statsmodels.api as sm
# from statsmodels.formula.api import ols

#parameters
m = 10 # out fluctuation-free source emits m photons in one sample interval 
N = 300 # number of samples taken

# Signal to Noise ration basic code in case useful later
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

# Function to get QE input from user and ensure it is a sensible value
def QEInput():
    global QE
    while True:
        try:
            QE = float(input("Enter Quantum Efficiency between 0 and 1: "))
            break
        except ValueError:
            print("Please enter a numerical value.") 
            continue
   
        
# Function to take the number of sensors and create an array of plots that will change size
# depening on number of sensors required.
def LoopPlots():
    global fig, axs
    i,j,k=0,0,0
    hists_per_row = 5
    hist_columns = math.ceil(len(df.columns)/hists_per_row)
    fig, axs = plt.subplots(hist_columns,hists_per_row)
    # fig.text(0.5, 0.04, 'Number of Photons Detected', ha='center', fontsize=16)
    # fig.text(0.04, 0.5, 'Photon Detection Probability', va='center', rotation='vertical', fontsize=16)
    for col in df.columns:
        if i < hist_columns-1:
            axs[i][j].set_xticklabels([])
        r = poisson.rvs(np.multiply(hist_n,df[col]).sum(), size=N)
        sns.distplot(r, ax=axs[i][j], hist=False, fit=norm, kde=False, color="r")
        axs[i][j].bar(hist_n, df[col], align='center', alpha=0.5)
        axs[i][j].set_title("Sensor "+str(k+1))
        # axs[i][j].set_ylim([0,(np.max(hist)*1.05)])
        axs[i][j].set_ylim([0,1])
        # axs[i][j].tick_params(axis=u'x', which=u'both',length=2)
        axs[i][j].axvline(np.multiply(hist_n,df[col]).sum(), linestyle='dashdot')
        # axs[i][j].text(np.mean(df[col])*0.1,0,str(np.mean(df[col])),rotation=90)
        k+=1
        j+=1
        if j%hists_per_row==0:
            i+=1
            j=0
    fig.supxlabel('Number of Photons Detected (D)', fontsize='large', fontweight='bold')
    fig.supylabel('Probability of D Photons being Detected', fontsize='large', fontweight='bold')
    plt.tight_layout()
    # Tidy x ticks to make them integers
    xint = np.arange(0,math.ceil(max(r)+1))
    plt.xticks(xint)
    plt.show()

# Function to interpolate the value of a specific photon count, to calculate residuals
# later in the code    
def prob_photon_num(count):
    interp1d = interpolate.interp1d(rLineValues[0],rLineValues[1],kind='linear')
    return interp1d(count)
    
# Get input from user on QE requested
QEInput()
# Ensure value falls between 0 and 1
while QE < 0 or QE > 1:
    print('This value is invalid. Please try again.')
    QEInput()
else:
    pass

# detector_events:
# This is a time series of photon detection events
# Each element corresponds to the time interval in which one photon was emitted from our fluctuation-free source
# Now we randomly select which of these photons reach the detector
# if detector_events is 1, then the photon was detected, if it is 0, then the photon was missed due to low QE

# Set up array of sensors where each one is the effective QE slice i.e up to 0.1. larger than 0.1 up to 0.2 etc.
while True:
    try:
      user_sensors = int(input("Enter number of sensors: "))
      break
    except ValueError:
        print("Please input integer only...") 
        continue
# User input determines numebr of sensors using the linspace below
QES = np.multiply(np.linspace(0,1, num = user_sensors+1),QE)
S = (user_sensors, N*m)
detector_events = np.zeros(S)
detector_counts = np.zeros((user_sensors,N)) # these are the accumulated photon counts in one sample interval
maxcounts = np.zeros(user_sensors)

# Create array of events based on random generation
q = 1 # Indice for while loop - starts on 1 to avoid index error when comparing 

rnd_num = np.random.rand(m*N) # Create random array for detection
while QES[q] <= QE:
    inds=np.where((rnd_num >= QES[q-1]) & (rnd_num < QES[q]))     # Select indices where criteria of detection are met     
    np.put(detector_events[q-1,:], inds, np.ones(np.size(inds))) # Add detection events based on ind
    if QES[q] == QE:
        break
    else:
        q = q+1


# Fill in count data based on events
t = 0
while (t < user_sensors):
    n = 0
    while n < N: # go through subsequent sample intervals
        # sum all photons within the current sample interval
        detector_counts[t,n] = np.sum(detector_events[t,n*m:(n+1)*m])
        n = n+1
    t = t+1

# Get max photons per sensor
t = 0
while (t < user_sensors): 
    maxcounts[t] = np.max(detector_counts[t]) # maximum number of photons received
    t = t + 1


# Get events for overall system
total_system_events = (rnd_num < QE).astype(int)
total_system_counts = np.zeros(N) # these are the accumulated photon counts in one sample interval

n = 0
while n < N: # go through subsequent sample intervals
    # sum all photons within the current sample interval
    total_system_counts[n] = np.sum(total_system_events[n*m:(n+1)*m])
    n = n+1

system_maxcounts = np.max(total_system_counts) # maximum number of photons received

# next we build a histogram
hist,hist_bins = np.histogram(total_system_counts, bins=np.arange(system_maxcounts+2))

total_hist = hist/N # normalise so that the values represent probabilities
total_hist_n = np.arange(system_maxcounts+1) # will be the x-axis for the plot, corresponds to number of photons detected

# save histogram
result = np.transpose([total_hist_n, total_hist]) # combine x and y axis in the right format for saving
np.savetxt("total_histogram.csv", result, delimiter=",") # save to csv file

# next we build the histograms
maxHist = int(np.max(maxcounts))
hist = np.zeros([user_sensors,maxHist+1])
hist_n = np.zeros([user_sensors,maxHist+1])
hist_bins = np.zeros([user_sensors,maxHist+2])
t = 0
while (t < user_sensors):
    hist[t],hist_bins[t] = np.histogram(detector_counts[t], bins=np.arange(np.max(maxcounts)+2))
    hist[t] = hist[t]/N # normalise so that the values represent probabilities
    hist_n = np.arange(maxHist+1) # will be the y-axis for the plot, corresponds to number of photons detected
    t = t + 1

# save histogram
df = pd.DataFrame(np.transpose(hist))
np.savetxt("histogram.csv", df, delimiter=",") # save to csv file

# Overall histogram for whole system as individual sensors' average
histA = [np.sum(hist[:,x]) for x in range(len(hist[0,:]))]
histA = [x/user_sensors for x in histA]

## Calculate Errors
xerr= (np.std(total_hist))

#### PLOTTING

# Set plot styles
plt.style.use('ggplot')
    
# Plot individual sensors. This is done by the else statement if 4 sensors or 
# less, or by the LoopPlots() function to dynamically change plot size
i,j,k=0,0,0
hists_per_row = 5

if user_sensors > hists_per_row:
    LoopPlots()
else:
    fig, axs = plt.subplots(1,hists_per_row)
    fig.text(0.5, 0.04, 'Number of Photons Detected', ha='center', fontsize=16)
    fig.text(0.04, 0.5, 'Photon Detection Probability', va='center', rotation='vertical', fontsize=16)
    for col in df.columns:
        r = poisson.rvs(np.multiply(hist_n,df[col]).sum(), size=N)
        sns.distplot(r, ax=axs[j], hist=False, fit=norm, kde=False, color="r")
        axs[j].bar(hist_n, df[col], align='center', alpha=0.5)
        axs[j].set_title("Sensor "+str(k+1))
        axs[j].set_ylim([0,1])
        if j == (user_sensors-1):
            break
        else:
            k+=1
            j+=1
    # Tidy x ticks to make them integers
    xint = np.arange(0,math.ceil(max(r)+1))
    plt.xticks(xint)
    plt.show()

# plot average histogram
plt.figure()
plt.title("Average probability for 1 sensor in full array with QE = " +  str(QE), fontsize=18)
plt.bar(hist_n, histA, align='center', alpha=0.5)
plt.xlabel("Number of detected photons (D)", fontsize=18)
plt.ylabel("Probability of detecting D photons", fontsize=16)
plt.show()

# plot total histogram
plt.figure()
plt.title("Overall probability for full array with QE = " +  str(QE), fontsize=20)
plt.bar(total_hist_n, total_hist, align='center', alpha=0.5)
plt.xlabel("Number of detected photons (D)", fontsize=18)
plt.ylabel("Probability of detecting D photons", fontsize=16)
plt.axvline(np.mean(total_system_counts), linestyle='dashdot')
plt.ylim([0,np.max((total_hist)+xerr)*1.1])
plt.show()

f, (h0, h1) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# Create Poisson distribution based on mean photons on new figure
# h0.figure()
# h0.title("Overall probability for full array with QE = " +  str(QE), fontsize=20)
# h0.xlabel("Number of detected photons (D)", fontsize=18)
# h0.ylabel("Probability of detecting D photons", fontsize=16)
r = poisson.rvs(np.mean(total_system_counts), size=N)
sns.distplot(r, hist=False, fit=norm, fit_kws={"color":"black"}, kde=False, color="r", label='Poisson Distribution for Same Mean', ax=h0)
h0.legend()
rLineValues = sns.distplot(r, hist=False, fit=norm, fit_kws={"color":"black"}, kde=False, color="r", label='Poisson Distribution for Same Mean', ax=h0).get_lines()[0].get_data()
xint = np.arange(0,math.ceil(max(r)+1))
h0.set_xticks(xint)

# Plot error bars of hist
col =[]
for i in range(0, len(total_hist_n)):
    if total_hist_n[i] == int(np.mean(total_system_counts)):
        col.append('magenta')  
    else:
        col.append('blue') 
for i in range (len(total_hist)):
    h0.errorbar(total_hist_n[i], total_hist[i], xerr, fmt = 'o',color = col[i], ecolor = 'lightgrey', elinewidth = 0, capsize=0)
h0.set_xticklabels(xint)
h0.set(xlabel="Number of detected photons (D)", ylabel="Probability of detecting D photons")

# Point for report - tried this but not suitable
# Use univariateSpline to form curve and find points for FWHM
# totalFit = interpolate.UnivariateSpline(total_hist_n, total_hist-np.max(total_hist)/2, s=0,k=3)
# r1, r2 = totalFit.roots()

### Change based on f/b 
linear_interp=interpolate.interp1d(total_hist_n,total_hist)
dummy_x_values = np.linspace(0,int(system_maxcounts),10000)
interp_data = [linear_interp(x) for x in dummy_x_values]
interp_data = [np.round(x,4) for x in interp_data]
half_max_value = np.max(interp_data)/2
half_max_values = np.where(interp_data == np.round(half_max_value,4))
h0.axvspan(dummy_x_values[half_max_values[0][0]], dummy_x_values[half_max_values[0][-1]], facecolor='g', alpha=0.5)


# Calculate residuals and plot below distribution comparison
totalCountResiduals = np.zeros(len(xint))
for i in range(0, len(xint)):
    if i < len(total_hist_n):
        totalCountResiduals[i] = total_hist[i] - prob_photon_num(i)
    else:
        totalCountResiduals[i] = 0 - prob_photon_num(i)
        
# Calculate sum of squares
sum_squares = np.round(np.sum(np.square(totalCountResiduals)),4)


h1.bar(xint, totalCountResiduals, color = 'mediumpurple')
# #988ed5 is a nice purple?
h1.set_xticks(xint)
h1.set(xlabel="Number of detected photons (D)", ylabel="Residuals")
h1.text(np.max(xint), np.max(totalCountResiduals)*0.5, 'RSS = ' + str(sum_squares), ha='center', fontsize=16)
f.suptitle("Overall probability for full array with QE = " +  str(QE), fontsize=20)
f.show()
# input('press <ENTER> to continue')





# %%
