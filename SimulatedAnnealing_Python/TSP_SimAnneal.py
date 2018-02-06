
# coding: utf-8

# ===================================================================
# Import necessary modules
# ===================================================================

import math
import time
import random
import copy
import sys

# For graphs
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================================================================
# Define Functions
# ===================================================================

# Calculates distance between two points (2-D points)
def calcDistance(a, b):
        distance = ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5 # 2-D distance
        return distance

# Calculates the square distance between two points (2-D points)
# Added as alternative to calcDistance. Change functions in code to use this method instead.
# This is a short-cut to avoid O(n**2/2) sqrt calls, which can be
# computationally expensive in terms of time
def calcDistanceSquared(a,b):
        distanceSquared = (a[0]-b[0])**2 + (a[1]-b[1])**2 # 2-D distance
        return distanceSquared

# Calculates the total distance of the trip
def GetTourLength(tour):
    l = 0
    n = len(tour)
    for i in range(0, n):
        l += calcDistance(nodelist[tour[i%(n-1)]], nodelist[tour[(i+1)%(n-1)]])
    return l

def readLine(inputFile):
        text = inputFile.readline().strip().split()
        textLength = len(text)
        return text[textLength-1]

# Initial tour generation function - generates a simple tour
def InitialTour(n):
    tour = []
    for i in range(0, n):
        tour.append(i)
    return tour

def acceptanceProbability(oldDist, newDist, t):
    # Acceptance Probability = e ^ [(oldDist - newDist) / t ]
    e = 2.71828 # Mathematical constant
    form1 = oldDist - newDist
    form2 = form1 / t
    if form2 >= 0:
        form2 = 0
    ap = e**form2
    return ap

# Swaps any 2 points in the tour with each other
def swap2(tour):
    n = range(len(tour))
    i1, i2 = random.sample(n, 2)
    tour[i1], tour[i2] = tour[i2], tour[i1]
    return tour

# ===================================================================
# Main Application
# ===================================================================

# Open input file
infile = open('st70.tsp', 'r')

# Read instance header
Name = infile.readline().strip().split()[1]
Comment = infile.readline().strip().split()[1]
Type = infile.readline().strip().split()[1]
Dimension = infile.readline().strip().split()[1]
EdgeWeightType = infile.readline().strip().split()[1]
infile.readline()

# Read node list
nodelist = []
N = int(Dimension)
for i in range(0, N):
    x,y = infile.readline().strip().split()[1:]
    nodelist.append([float(x), float(y)])

# Close input file
infile.close()

#print nodelist

# Create benchmark through optimal tour file
optFile = open('st70.opt.tour', 'r')

# Next, read the optimal solution file
optName = optFile.readline().strip().split()[2] # NAME
optFileType = optFile.readline().strip().split()[1] # TYPE
optDataSize = optFile.readline().strip().split()[2] # DIMENSION
optDataDef = optFile.readline() # TOUR_SECTION
optPath = []
optTrip = []
for i in range(0, int(optDataSize)):
        ind = optFile.readline().strip().split()[0]
        optPath.append(int(ind))
optFile.close()

# Normalizes the optimal path to match the nodelist of the .tsp file
optPath[:] = [x - 1 for x in optPath]

# Construct the optimal trip
# Uses same distance formula as other paths for consistency
optTrip = GetTourLength(optPath)

# Debug to print trip and distance
#print optTrip
#print optPath

# Setup variables
tour = InitialTour(N)
distance = GetTourLength(tour)
firstDist = distance
firstTour = tour
firstSwapTour = tour

# Initializes randomized tour for simulated annealing
# Randomizing is done by performing one 2swap for the length of the array - 1
# Minus 1 is added to reduce chance of reverting back to original arrays in small tours.

# Generate Simulated Annealing starting point
for i in range(len(firstTour) - 1):
    startTour = swap2(firstTour)

# Generate Local Neighborhood Search only starting point
startSwapTour = startTour

# Debugging Swap and Length functions

startDistance = GetTourLength(startTour)
startSwapDistance = GetTourLength(startSwapTour)

# Debugging with print
#print("Starting SA tour length: " + str(startDistance))
#print startTour

#print("Starting 2swap tour length: " + str(startSwapDistance))
#print startSwapTour

# Debug to make sure nodelist not deleted
#print nodelist

# 2swap Only

# Start the algorithm timer
startswaptime = time.time()
# 10k instances to test algorithm
swapIterationCount = 110000
swapCount = 0
localTour = startSwapTour # Reinitializes tour
localDistance = startSwapDistance # Reinitializes distance

for i in range(swapIterationCount):
    swap2tour = swap2(localTour)
    zValue = GetTourLength(startTour)
    if zValue < localDistance:
        swapCount += 1
        #print("Tour " + str(swapCount) + " length: " + str(zValue))
        #print swap2tour
        localDistance = zValue
        localTour = swap2tour
    swap2tour = localTour

#Stops and records time
endswaptime = time.time()
algoswaptime = endswaptime - startswaptime

# Pseudocode - 2opt
#Iterations = Value > 0
#localTour = startSwapTour # Reinitializes starting tour
#localDistance = startSwapDistance # Reinitializes distance

# Searches for best solution within n iterations
#for i in range(Iterations):
    #swap2tour = swap2(localTour)
    #zValue = GetTourLength(startTour)
    #if zValue < localDistance:
        #localDistance = zValue
        #localTour = swap2tour
    #swap2tour = localTour

# Simulated Annealing

# Start the algorithm timer
starttime = time.time()

# Setup variables
startTour2 = startTour
oldDist = GetTourLength(startTour)
Temp = 1000.0 # Temperature (Result Printing)
t = Temp # to be variable in algorithm
tMin = 0.01 # Low value
alpha = 0.9 # Between 0.8 and 0.99
counter = 0
passcount = 0
highStart = 10000000000 # Extremely high number to catch any value for first optimal
iterationsTemp = 1000

iterationDistResults = []
optDistResults = []

# Perform simulated annealing

while t > tMin:

    i = 1
    while i <= iterationsTemp:
        newTour = swap2(startTour2)
        newDist = GetTourLength(newTour)
        #print newTour # Comment to turn off printing
        #print newDist # Comment to turn off printing
        ap = acceptanceProbability(oldDist, newDist, t)
        if newDist < oldDist:
            startTour2 = newTour
            oldDist = newDist
        elif ap > random.random():
            #print "Accept value" # Comment to turn off printing
            startTour2 = newTour
            oldDist = newDist
            #print oldDist # Comment to turn off printing
            #print startTour2 # Comment to turn off printing
        if newDist < highStart:
            highStart = newDist
            optTour = newTour
            optDist = newDist
            optDistResults.append(optDist)
        i += 1
        counter = counter + 1
        swapDist = newDist
        iterationDistResults.append((oldDist, swapDist))
    t = t*alpha
    passcount = passcount + 1
    #print ("Iteration Round " + str(passcount) + " Done") # Comment to turn off printing

print ("")
print ("Simulated Annealing Done")

#Stops and records time
endtime = time.time()
algotime = endtime - starttime

# Simulated Annealing Pseudocode

# Setup variables
#startTour2 = startTour # Reinitialize tour
#oldDist = GetTourLength(startTour) # Reinitialize distance
#Temp = 1000.0 # Temperature
#t = Temp # to be variable in algorithm
#tMin = 0.01 # Low value
#alpha = 0.8 # Between 0.8 and 0.99
#highStart = 10000000000 # Extremely high number to catch any value for first optimal
#iterationsTemp = 1000

# Perform simulated annealing

#while t > tMin:

    #i = 1
    #while i <= iterationsTemp:
        #newTour = swap2(startTour2)
        #newDist = GetTourLength(newTour)
        #ap = acceptanceProbability(oldDist, newDist, t)
        #if newDist < oldDist:
            #startTour2 = newTour
            #oldDist = newDist
        #elif ap > random.random():
            #startTour2 = newTour
            #oldDist = newDist
        #if newDist < highStart:
            #highStart = newDist
            #optTour = newTour
            #optDist = newDist
        #i += 1
    #t = t*alpha

# Converts the values into dataframes
df_iter = pd.DataFrame(iterationDistResults)
df_iter.columns = ['LastAcceptedValue', 'SwapsValue']

# Adds IDs to the front of the list for graphing
df_iter.insert(0, 'Iteration', range(1, 1 + len(df_iter)))

# Converts the values into dataframes
df_opt = pd.DataFrame(optDistResults)
df_opt.columns = ['LowestOptimal']

# Adds IDs to the front of the list for graphing
df_opt.insert(0, 'Step', range(1, 1 + len(df_opt)))

# Shows dataframe in Jupyter Notebook environment
#df_iter # Comment on/off to turn on viewing in Jupyter

# Shows dataframe in Jupyter Notebook environment
#df_opt # Comment on/off to turn on viewing in Jupyter

# Creates plot graph of Lowest Accepted Value change over time

df_iter.set_index('Iteration').plot()
plt.show()


# Creates plot graph of Lowest Accepted Value change over time

df_opt.set_index('Step').plot()
plt.show()

# Collection of print statements after algorithm
print("")
print("== Overview ==")
print("")
print("Instance Name: " + str(Name))
print("Instance Size: " + str(N))
print("")
print("-- 2opt Overview --")
print("")
print("Starting 2opt Distance: " + str(startSwapDistance))
print("2opt Iterations: " + str(swapIterationCount))
print("2opt Solution Distance: " + str(localDistance))
#print("2swap Solution Path: " + str(localTour)) # Takes up space, so comment on/off for viewing
print("CPU Time (2swap): " +str(algoswaptime) + " seconds")
print("")
print("-- Simulated Annealing Overview --")
print("")
print("Starting SA Distance: " + str(startDistance))
print("Temperature Start: " + str(Temp))
print("Temperature End: " + str(tMin))
print("Alpha (Change): " + str(alpha))
print("Iterations per Temp: " + str(iterationsTemp))
print("SA Swap Iterations: " + str(counter))
print("Solution Distance: " + str(optDist))
#print("Solution Path: " + str(optTour)) # Takes up space, so comment on/off for viewing
print("CPU Time (SA): " +str(algotime) + " seconds")
print("")
print("-- Optimal Values --")
print("")
print("Optimal Distance: " + str(optTrip))
#print("Optimal Path: " + str(optPath)) # Takes up space, so comment on/off for viewing
