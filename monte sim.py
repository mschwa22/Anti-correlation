# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 16:18:37 2020

@author: Maddy
"""

# Want to create a Monte Carlo simulation to test the reliablness of the DIB 7030 (to start with).
# The possible approach to doing this is to take the stdev of the EW of the DIB, to make that the 
# bounds of which the random number generator will pick a number from. I'm doing it this way so that
# it will mimic a Gaussian distr. in which the data varies.

# After generating a random number, it will be added (or subtracted depending on the sign of the number)
# to the already existing, original data to create a new, fake, data set to use. With this new data set
# I will calculate the pearson correlation coefficient that it has with all of the other DIBs that are 
# used in the previous file (the 65 other DIBs)

# This process will be repeated about 100 times to test how much the data varies depending on the fake 
# data. One way to test how reliable the 7030 is, is to calculate the relative error of the fake data 
# correlation. So for each correlation, the expected value is the original real data correlations, and 
# then the measured value is the fake data. Do that for each correlation, take the average of it so that
# we get one number for each fake data set. Then after all 100 times this is done, take the average of 
# all the numbers taken from each set of fake data so that we can get an overall error percentage of how 
# varying the measurements are. So basically it would be (sum of each fake average error)/100