# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:07:08 2020

@author: maddy
"""
import pandas as pd

DIB_data = pd.read_excel('DIB Measurements for APO Catalog.xlsx', index_col = None)
sight_data = pd.read_excel('Info on sight lines.xlsx', index_col = 0).T

# Replace unnamed column headers with the names of the next line. 
new_header = DIB_data.iloc[0]
DIB_data.columns = new_header
DIB_data.drop([0], axis = 0)


units = input("Would you like to keep the units? ")
if units.lower() == "no":
    DIB_data = DIB_data.drop([0, 1], axis = 0)
    
    
# Get rid of any columns that you do not want. Will ask for an input for the following columns that you would like to delete.
# 'idx'      - the index of the DIB
# 'Wav'      - The wavlength of the DIB
# 'FWHM'     - The full width - half maximum of the DIB. (Given both in units of km/s and A)
# 'xm2'      - Given in km/s and A. (Sidenote: I am not sure what this actually is)
# 'Mea Wav'  - The mean wavlength
# 'Mea FWHM' - The mean full width - half maximum
# 'UL'       - (I don't really know what this is)
# 'Mea EW'   - The mean equivalent width
# 'Mea Err'  - The mean error for the equivalent width
# 'Result'   - The result of the recorded data. (whether the data is good/usable or not)

unwanted_columns = input("What columns would you like to get rid of? (use ',' only to seperate columns, not spaces) ")
unwanted_columns = unwanted_columns.split(',')
DIB_data = DIB_data.drop(unwanted_columns, axis = 1)

# Make the data more easily available to use by splitting it up.
Result = DIB_data['Result']
Mean_EW = DIB_data['Mea EW']
Mean_Error = DIB_data['Mea Err']
DIBs = DIB_data[['idx', 'Wav']]
# Rename columns of EW, Error, and Result to match the sight data columns
Mean_EW.columns = sight_data.columns
Mean_Error.columns = sight_data.columns
Result.columns = sight_data.columns

# Want to use only the data that has the result "D". In previous version, found the indice at which a 'D' occurs, then used the same index
# to find the corrleating EW and error with this 'D' result. The problem with this method is it can get very confusing with all of the
# indexing. It is very easy to get overwhelmed and lost with this.
# Option 1 - loop through each row in Result and count how many times 'D' appears for that row ( /25). Keep track of that number in a list.
#            This could work in determining the DIBs that have the most useable data out of the set if that is how we want to find the top
#            10 DIBs. 
# Option 2 - Do what I did before with all of the indexing.

# Will use Option 1

# Get a user input on the method they would like to use to select the top DIBs
# 'Max'   - Will select the DIBS that have the most amount of data sets out of 25.
# 'Error' - Will select the DIBs that have the smallest errors
#
method = input('Enter the method in which you would like to select the DIBs (Max, Error, etc.) ')

if method.lower() == 'max':
    count = []
    for row in range(len(Result)):
        count.append(Result.iloc[row].str.count('D').sum())
    
    # Add the data count list to the DIB information containing their wavelength and index.
    DIBs.insert(2, "Data Count", count, False)
    # Sort the DIBs from most data to least
    DIBs = DIBs.sort_values(by = 'Data Count', ascending = False)
    
    # Get the number of DIBs that are to be compared
    amount = input('How many DIBs would you like to compare? ')
    
    
    # Normalize the data
    normalize = input('Would you like to normalize the data? ')
    if normalize.lower() == 'yes':
        Mean_EW_Norm = Mean_EW.div(sight_data.iloc[0])
        
        Mean_EW_Norm.insert(25, 'Data Count', count, False)
        Mean_EW_Norm = Mean_EW_Norm.sort_values(by = 'Data Count', ascending = False)
        
        # Find correlation between the wanted amount of DIBs
        pears_corr = Mean_EW_Norm[0:int(amount)].corr(method = 'pearson')
    
        
