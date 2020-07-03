# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:07:08 2020

@author: maddy
"""
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

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
Result = DIB_data['Result'].astype('str')
Mean_EW = DIB_data['Mea EW']
Mean_Error = DIB_data['Mea Err']
DIBs = DIB_data[['idx', 'Wav']]
# Rename columns of EW, Error, and Result to match the sight data columns
Mean_EW.columns = sight_data.columns
Mean_Error.columns = sight_data.columns
Result.columns = sight_data.columns

# Rounds any number to two decimal places
DIBs['Wav'] = DIBs['Wav'].astype(np.float64)
DIBs = DIBs.round(2)


# To ensure that we are only using data that has a result 'D', find any result that is not that, and replace the value with np.nan
n = 0
while n < 25:
    for ind in range(2,559):
        if Result.loc[ind, Result.columns[n]] != 'D':
            Mean_EW.at[ind, Result.columns[n]] = np.nan
            
    n += 1


# Want to use only the data that has the result "D". In previous version, found the indice at which a 'D' occurs, then used the same index
# to find the corrleating EW and error with this 'D' result. The problem with this method is it can get very confusing with all of the
# indexing. It is very easy to get overwhelmed and lost with this.
# Option 1 - loop through each row in Result and count how many times 'D' appears for that row ( /25). Keep track of that number in a list.
#            This could work in determining the DIBs that have the most useable data out of the set if that is how we want to find the top
#            10 DIBs. 
# Option 2 - Do what I did before with all of the indexing.

# Will use Option 1

# Get a user input on the method they would like to use to select the top DIBs
# 'Max'    - Will select the DIBS that have the most amount of data sets out of 25.
# 'Select' - Will allow for a specific selection of DIBs. The two select options are: You can select specific DIBs based off of its wavelength,
#            and the other is the minimum amount of data sets each DIB has. The most data sets a DIB can is 25 and the minimum is 0. If 20 is 
#            the chosen number of data sets, all DIBs that have 20 or more sets of data will be compared to each other.
#

# Normalize the data
normalize = input('Would you like to normalize the data? ')
if normalize.lower() == 'yes':
    Mean_EW_Norm = Mean_EW.div(sight_data.iloc[0])
 
       
method = input('Enter the method in which you would like to select the DIBs (Max, Select) ')

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
    
    
    if normalize.lower() == 'yes':
        
        Mean_EW_Norm.insert(25, 'Data Count', count, False)
        Mean_EW_Norm = Mean_EW_Norm.sort_values(by = 'Data Count', ascending = False)
        
        # Find correlation between the wanted amount of DIBs
        Mean_EW_Norm = Mean_EW_Norm.astype(np.float64)
        pears_corr = Mean_EW_Norm[0:int(amount)].T.corr(method = 'pearson')
        
        title = 'Normalized Data Pearson Correlation'
    
    elif normalize.lower() == 'no':
        
        Mean_EW.insert(25, 'Data Count', count, False)
        Mean_EW = Mean_EW.sort_values(by = 'Data Count', ascending = False)
        
        # Find correlation between the wanted amount of DIBs
        Mean_EW = Mean_EW.astype(np.float64)
        pears_corr = Mean_EW[0:int(amount)].T.corr(method = 'pearson')
        
        title = 'Non-normalized Data Pearson Correlation'

    # Create a list of the wavelengths of the used DIBs. Then create a heatmap to visually display the data correlations
    labels = list(DIBs['Wav'].iloc[0:int(amount)])
    sb.heatmap(pears_corr, xticklabels = labels, yticklabels = labels, cmap='viridis', annot = True, linewidth = 0.5, annot_kws = {'fontsize': 6}).set(title = title)
    
elif method.lower() == 'select':
    
    selection = input('How would you like to select the DIBs? (Wavelength, amount of data sets) ')
    
    if selection.lower() == 'amount of data sets':
        count = []
        for row in range(len(Result)):
            count.append(Result.iloc[row].str.count('D').sum())
    
        # Add the data count list to the DIB information containing their wavelength and index.
        DIBs.insert(2, "Data Count", count, False)
        # Sort the DIBs from most data to least
        DIBs = DIBs.sort_values(by = 'Data Count', ascending = False)
        
        # Find the minimum amount of data sets /25 that are to be compared
        min_data = input('What is the minimum amount of data sets you would like to compare? (out of 25) ')
        
        if normalize.lower() == 'yes':
           
            Mean_EW_Norm.insert(25, 'Data Count', count, False)
            Mean_EW_Norm = Mean_EW_Norm.sort_values(by = 'Data Count', ascending = False)
            
            Mean_EW_Norm = Mean_EW_Norm.astype(np.float64)
            pears_corr = Mean_EW_Norm.loc[Mean_EW_Norm['Data Count'] >= int(min_data)].T.corr(method = 'pearson')
            
            title = 'Normalized Data Pearson Correlation'
            
        elif normalize.lower() == 'no':
            
            Mean_EW.insert(25, 'Data Count', count, False)
            Mean_EW = Mean_EW.sort_values(by = 'Data Count', ascending = False)
            
            Mean_EW = Mean_EW.astype(np.float64)
            pears_corr = Mean_EW.loc[Mean_EW['Data Count'] >= int(min_data)].T.corr(method = 'pearson')
            
            title = 'Non-normalized Data Pearson Correlation'

        
        # Create a list of the wavelengths of the used DIBs. Then create a heatmap to visually display the data correlations
        labels = list(DIBs['Wav'].loc[DIBs['Data Count'] >= int(min_data)])
        fig, ax = plt.subplots(figsize = (15,10))
        sb.heatmap(pears_corr, xticklabels = labels, yticklabels = labels, cmap='viridis', ax = ax).set(title = title)
        plt.tick_params(axis = 'x', labelsize = 10)
        plt.tick_params(axis = 'y', labelsize = 10)
        plt.show()
        
        # Create a histogram of the data
        hist, bin_edges = np.histogram(pears_corr, bins = 20)
        plt.bar(bin_edges[:-1], hist/2, width = 0.075, alpha = 0.7, edgecolor = 'black')
        plt.xlim(min(bin_edges), max(bin_edges))
        plt.ylabel('Frequency')
        plt.xlabel('Pearson Correlation')
        plt.title('Pearson Correlation as a Function of Frequency')
        plt.show()
        
        
    elif selection.lower() == 'wavelength':
        
        wavelengths = input('Please enter the wavlength (without the decimals) of wanted DIBs. ')
        wavelengths = wavelengths.strip(' ').split(',')
        wavelengths = [int(i) for i in wavelengths]
        
        # Get rid of decimal points in the wavelength column so that it is easier to find a specific DIB by wavelength.
        DIBs['Wav'] = DIBs['Wav'].apply(np.floor).astype(np.int64)
        
        spec_DIBs = pd.DataFrame(columns = Mean_EW.columns)
        
        # Find index of the wanted DIBs based off of the wavelength given
        for wave in wavelengths:
            indeces = DIBs[DIBs['Wav'] == wave].index.values
            
            if normalize.lower() == 'yes':
                
                # Add DIB data to the already existing dataframe in order to get the correlation.
                data = Mean_EW_Norm.iloc[indeces - 2]
                spec_DIBs = spec_DIBs.append(data)
                
            elif normalize.lower() == 'no':
                
                data = Mean_EW.iloc[indeces - 2]
                spec_DIBs = spec_DIBs.append(data)
        
        spec_DIBs = spec_DIBs.astype(np.float64)
        
        pears_corr = spec_DIBs.T.corr(method = 'pearson')
        #sb.heatmap(pears_corr, xticklabels = wavelengths, yticklabels = wavelengths, cmap='viridis', annot = True, linewidth = 0.5, annot_kws = {'fontsize': 6})
        fig, ax = plt.subplots(figsize = (15,10))
        sb.heatmap(pears_corr, xticklabels = wavelengths, yticklabels = wavelengths, cmap='viridis', ax = ax)