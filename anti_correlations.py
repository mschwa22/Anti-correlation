# -*- coding: utf-8 -*-
"""
Created on Sun May 24 12:36:20 2020

@author: maddy
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from collections import Counter

# This will take in all the data for the DIBs with their respective lines of sight, get rid of any data that isnt required, and then compare the good data
# with each other in a graph to determine if there are any anti-correlations between the DIBs

file_name = 'DIB Measurements for APO Catalog v2.csv'

# Import DIB data
import csv
f = open(file_name)
csv_f = csv.reader(f)

# Create holding places for all the data for each line of sight
DIB_num = []

Mean_EW = [ [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]        
Mean_Error = [ [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
Result = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]


# Make lists of the required data

i = 12      # Indices to be run through to get all data
k = 9
t = 10

for row in csv_f:
    DIB_num.append(row[0])
        
    Mean_EW[0].append(row[k])
    Mean_Error[0].append(row[t])
    Result[0].append(row[i])
        
    Mean_EW[1].append(row[k + 7])
    Mean_Error[1].append(row[t + 7])
    Result[1].append(row[i + 7])
    
    Mean_EW[2].append(row[k + 14])
    Mean_Error[2].append(row[t + 14])
    Result[2].append(row[i + 14])
    
    Mean_EW[3].append(row[k + 21])
    Mean_Error[3].append(row[t + 21])
    Result[3].append(row[i + 21])
    
    Mean_EW[4].append(row[k + 28])
    Mean_Error[4].append(row[t + 28])
    Result[4].append(row[i + 28])
    
    Mean_EW[5].append(row[k + 35])
    Mean_Error[5].append(row[t + 35])
    Result[5].append(row[i + 35])
    
    Mean_EW[6].append(row[k + 42])
    Mean_Error[6].append(row[t + 42])
    Result[6].append(row[i + 42])
    
    Mean_EW[7].append(row[k + 49])
    Mean_Error[7].append(row[t + 49])
    Result[7].append(row[i + 49])
    
    Mean_EW[8].append(row[k + 56])
    Mean_Error[8].append(row[t + 56])
    Result[8].append(row[i + 56])
    
    Mean_EW[9].append(row[k + 63])
    Mean_Error[9].append(row[t + 63])
    Result[9].append(row[i + 63])
    
    Mean_EW[10].append(row[k + 70])
    Mean_Error[10].append(row[t + 70])
    Result[10].append(row[i + 70])
    
    Mean_EW[11].append(row[k + 77])
    Mean_Error[11].append(row[t + 77])
    Result[11].append(row[i + 77])
    
    Mean_EW[12].append(row[k + 84])
    Mean_Error[12].append(row[t + 84])
    Result[12].append(row[i + 84])
    
    Mean_EW[13].append(row[k + 91])
    Mean_Error[13].append(row[t + 91])
    Result[13].append(row[i + 91])
    
    Mean_EW[14].append(row[k + 98])
    Mean_Error[14].append(row[t + 98])
    Result[14].append(row[i + 98])
    
    Mean_EW[15].append(row[k + 105])
    Mean_Error[15].append(row[t + 105])
    Result[15].append(row[i + 105])
    
    Mean_EW[16].append(row[k + 112])
    Mean_Error[16].append(row[t + 112])
    Result[16].append(row[i + 112])
    
    Mean_EW[17].append(row[k + 119])
    Mean_Error[17].append(row[t + 119])
    Result[17].append(row[i + 119])
    
    Mean_EW[18].append(row[k + 126])
    Mean_Error[18].append(row[t + 126])
    Result[18].append(row[i + 126])
    
    Mean_EW[19].append(row[k + 133])
    Mean_Error[19].append(row[t + 133])
    Result[19].append(row[i + 133])
    
    Mean_EW[20].append(row[k + 140])
    Mean_Error[20].append(row[t + 140])
    Result[20].append(row[i + 140])
    
    Mean_EW[21].append(row[k + 147])
    Mean_Error[21].append(row[t + 147])
    Result[21].append(row[i + 147])
    
    Mean_EW[22].append(row[k + 154])
    Mean_Error[22].append(row[t + 154])
    Result[22].append(row[i + 154])
    
    Mean_EW[23].append(row[k + 161])
    Mean_Error[23].append(row[t + 161])
    Result[23].append(row[i + 161])
    
    Mean_EW[24].append(row[k + 168])
    Mean_Error[24].append(row[t + 168])
    Result[24].append(row[i + 168])

    
# Remove excess data (empty spots, titles)
del DIB_num[:3], DIB_num[557:]
for m in range(len(Mean_EW)):
    
    del Mean_EW[m][:3], Mean_EW[m][557:]
    del Mean_Error[m][:3], Mean_Error[m][557:]
    del Result[m][:3], Result[m][557:]


# Make a list of indices where the result is 'D' for each DIB in each line of sight

good_result = [ [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]

for s in range(len(good_result)):
    for n in range(len(Result[s])):
        if Result[s][n] == 'D':
            good_result[s].append(n)
        


# Find the DIB that have the lowest error where the result == 'D'. 
Good_Mean_Error = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
for index in range(len(good_result)):
    for position in range(len(good_result[index])):
        Good_Mean_Error[index].append(float(Mean_Error[index][good_result[index][position]]))
        
Good_Mean_Error = np.array(Good_Mean_Error)
good_result = np.array(good_result)
  


# Count how many data sets each DIB has.
def CountList(lst, var):
    
    return sum(var in item for item in lst)


count = []
for DIB in range(len(DIB_num)):
    
    count.append(CountList(good_result, DIB))

# Create a list of the top DIB numbers/indeces by amount of line of sight data.
index_list = []
for ind in range(len(count)):
    if count[ind] == 25:
        index_list.append(ind)
    
for ind in range(len(count)):
    if count[ind] == 24:
        index_list.append(ind)
        
for ind in range(len(count)):
    if count[ind] == 23:
        index_list.append(ind)
       
# Create a list of the top DIBs and how many lines of sight data they each have
final_list = []
for i in range(10):
    max1 = 0
    for k in range(len(count)):
        if count[k] > max1:
            max1 = count[k]
            
    count.remove(max1)
    final_list.append(max1)     


# Get a list of the top 10 DIBs that have the most lines of sight and the data from each 
num = 0  
x_DIB = [[],[],[],[],[],[],[],[],[],[]]    
while num < 10:
    for w in range(len(good_result)):
        for b in range(len(good_result[w])):
            if good_result[w][b] == index_list[num]:
            
                x_DIB[num].append(float(Mean_EW[w][index_list[num]]))
            
    num += 1
    
x_DIB = np.array(x_DIB)
# print(np.corrcoef(x_DIB[0], x_DIB[1]))


# Get the Pearson correlation coefficient for the data set of x and y. Want to check if the DIBs have the same number of lines of sight.
# If they do, compare them directly using the correlation coefficient, if not, find the missing line of sight(s) from the smaller DIB
# data set and remove the data from the same line of sight thats missing in the larger DIB data set. This ensures that when comparing 
# the two DIBs, the same lines of sight are being used for each DIB, therefore it is a fair comparison.

for DIB in range(num-1):
    for h in range(1, num):
        if len(x_DIB[DIB]) == len(x_DIB[h]):
            print('Correlation between DIB', DIB_num[index_list[DIB]], 'and', DIB_num[index_list[h]], 'is', scipy.stats.pearsonr(x_DIB[DIB], x_DIB[h])[0])
        elif len(x_DIB[DIB]) > len(x_DIB[h]):
            sight_1 = []
            sight_2 = []
            for i, lst in enumerate(good_result):
                for j, sght in enumerate(lst):
                    if sght == index_list[DIB]:
                        sight_1.append(i)
                    elif sght == index_list[h]:
                        sight_2.append(i)
            diff = list(set(sight_1).difference(set(sight_2)))
            miss = []
            #for g in range(len(diff))                      # This is commented out because it is not working the way I would like and therefore is messing with the data
                #miss.append(x_DIB[DIB].pop(diff[g]))
            #print('Correlation between DIB', DIB_num[index_list[DIB]], 'and', DIB_num[index_list[h]], 'is', scipy.stats.pearsonr(x_DIB[DIB], x_DIB[h])[0])