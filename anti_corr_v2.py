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
            Mean_Error.at[ind, Result.columns[n]] = np.nan
            
    n += 1


# Want to use only the data that has the result "D". In previous version, found the indices at which a 'D' occurs, then used the same index
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
# 'Error'  - Will select DIBs based off of their relative uncertainty. All sightlines for each DIB must be <= the uncertainty percantage that 
#            is chosen. Even if one sightline is over the limit, that DIB will not be counted. NOTE: This method is not great because it does 
#            not leave many DIBs that leave a good amount of usable and reliable data. Do not recommend using this method

# Create an upper limit and lower limit EW using the mean error.
Mean_EW_uplim = Mean_EW.add(Mean_Error)
Mean_EW_lowlim = Mean_EW.sub(Mean_Error)

# Normalize the data
normalize = input('Would you like to normalize the data? ')
if normalize.lower() == 'yes':
    Mean_EW_Norm = Mean_EW.div(sight_data.iloc[0])
    Mean_EW_uplim_Norm = Mean_EW_uplim.div(sight_data.iloc[0])
    Mean_EW_lowlim_Norm = Mean_EW_lowlim.div(sight_data.iloc[0])
 
       
method = input('Enter the method in which you would like to select the DIBs (Max, Select, Error) ')

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
        pears_corr = Mean_EW_Norm[0:int(amount)].iloc[:, :-1].T.corr(method = 'pearson')
        
        title = 'Normalized Data Pearson Correlation'
    
    elif normalize.lower() == 'no':
        
        Mean_EW.insert(25, 'Data Count', count, False)
        Mean_EW = Mean_EW.sort_values(by = 'Data Count', ascending = False)
        
        # Find correlation between the wanted amount of DIBs
        Mean_EW = Mean_EW.astype(np.float64)
        pears_corr = Mean_EW[0:int(amount)].iloc[:, :-1].T.corr(method = 'pearson')
        
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
            pears_corr = Mean_EW_Norm.loc[Mean_EW_Norm['Data Count'] >= int(min_data)].iloc[:, :-1].T.corr(method = 'pearson')
            
            title = 'Normalized Data Pearson Correlation'
            
        elif normalize.lower() == 'no':
            
            Mean_EW.insert(25, 'Data Count', count, False)
            Mean_EW = Mean_EW.sort_values(by = 'Data Count', ascending = False)
            
            Mean_EW = Mean_EW.astype(np.float64)
            pears_corr = Mean_EW.loc[Mean_EW['Data Count'] >= int(min_data)].iloc[:, :-1].T.corr(method = 'pearson')
            
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
        spec_DIBs_uplim = pd.DataFrame(columns = Mean_EW.columns)
        spec_DIBs_lowlim = pd.DataFrame(columns = Mean_EW.columns)
        
        # Find index of the wanted DIBs based off of the wavelength given
        for wave in wavelengths:
            indeces = DIBs[DIBs['Wav'] == wave].index.values
            
            if normalize.lower() == 'yes':
                
                # Add DIB data to the already existing dataframe in order to get the correlation.
                data = Mean_EW_Norm.iloc[indeces - 2]
                spec_DIBs = spec_DIBs.append(data)
                
                data_uplim = Mean_EW_uplim_Norm.iloc[indeces - 2]
                spec_DIBs_uplim = spec_DIBs_uplim.append(data_uplim)
                
                data_lowlim = Mean_EW_lowlim_Norm.iloc[indeces - 2]
                spec_DIBs_lowlim = spec_DIBs_lowlim.append(data_lowlim)
                
                title = 'Selected DIBs with Normalized Data for Pearson Correlation'
                
            elif normalize.lower() == 'no':
                
                data = Mean_EW.iloc[indeces - 2]
                spec_DIBs = spec_DIBs.append(data)
                
                data_uplim = Mean_EW_uplim.iloc[indeces - 2]
                spec_DIBs_uplim = spec_DIBs_uplim.append(data_uplim)
                
                data_lowlim = Mean_EW_lowlim.iloc[indeces - 2]
                spec_DIBs_lowlim = spec_DIBs_lowlim.append(data_lowlim)
                
                title = 'Selected DIBs with Raw Data for Pearson Correlation'
        
        spec_DIBs = spec_DIBs.astype(np.float64)
        spec_DIBs_uplim = spec_DIBs_uplim.astype(np.float64)
        spec_DIBs_lowlim = spec_DIBs_lowlim.astype(np.float64)
        
        #waves = [4501,5780,5797,6379,6284,6613,7224,6270,5850,6353,6792,5789,4963,4984]
        
        pears_corr = spec_DIBs.T.corr(method = 'pearson')
        
        # To better organize the correlation data, take the average r values for each DIB and display them from highest to lowest ascending values.
        pears_corr['average'] = pears_corr.mean(axis = 1)
        pears_corr = pears_corr.sort_values(by = 'average', ascending  = False)
        
        waves = list(DIBs['Wav'].iloc[pears_corr.index - 2])
        #sb.heatmap(pears_corr, xticklabels = waves, yticklabels = waves, cmap='viridis', annot = True, linewidth = 0.5, annot_kws = {'fontsize': 6}).set(title = title)
        fig, ax = plt.subplots(figsize = (15,10))
        sb.heatmap(pears_corr.iloc[:, :-1], xticklabels = wavelengths, yticklabels = waves, cmap='viridis', ax = ax).set(title = title)
        plt.show()
        
        # Create a histogram of the data 
        hist, bin_edges = np.histogram(pears_corr, bins = 20)
        plt.bar(bin_edges[:-1], hist/2, width = 0.075, alpha = 0.7, edgecolor = 'black')
        plt.xlim(min(bin_edges), max(bin_edges))
        plt.ylabel('Frequency')
        plt.xlabel('Pearson Correlation')
        plt.title('Pearson Correlation as a Function of Frequency')
        plt.show()
        
        # Plot for the upper limit data
        #pears_corr = spec_DIBs_uplim.T.corr(method = 'pearson')
        #sb.heatmap(pears_corr, xticklabels = wavelengths, yticklabels = wavelengths, cmap='viridis', annot = True, linewidth = 0.5, annot_kws = {'fontsize': 6}).set(title = 'Upper Limit Normalized Data for Correlation')
        #fig, ax = plt.subplots(figsize = (15,10))
        #sb.heatmap(pears_corr, xticklabels = wavelengths, yticklabels = wavelengths, cmap='viridis', ax = ax).set(title = title)
        #plt.show()
        
         # Create a histogram of the data 
        #hist, bin_edges = np.histogram(pears_corr, bins = 20)
        #plt.bar(bin_edges[:-1], hist/2, width = 0.075, alpha = 0.7, edgecolor = 'black')
        #plt.xlim(min(bin_edges), max(bin_edges))
        #plt.ylabel('Frequency')
        #plt.xlabel('Pearson Correlation')
        #plt.title('Upper Limit Pearson Correlation as a Function of Frequency')
        #plt.show()
        
        #pears_corr = spec_DIBs_lowlim.T.corr(method = 'pearson')
        #sb.heatmap(pears_corr, xticklabels = wavelengths, yticklabels = wavelengths, cmap='viridis', annot = True, linewidth = 0.5, annot_kws = {'fontsize': 6}).set(title = 'Lower Limit Normalized Data for Correlation')
        #fig, ax = plt.subplots(figsize = (15,10))
        #sb.heatmap(pears_corr, xticklabels = wavelengths, yticklabels = wavelengths, cmap='viridis', ax = ax).set(title = title)
        #plt.show()
        
        # Create a histogram of the data 
        #hist, bin_edges = np.histogram(pears_corr, bins = 20)
        #plt.bar(bin_edges[:-1], hist/2, width = 0.075, alpha = 0.7, edgecolor = 'black')
        #plt.xlim(min(bin_edges), max(bin_edges))
        #plt.ylabel('Frequency')
        #plt.xlabel('Pearson Correlation')
        #plt.title('Lower Limit Pearson Correlation as a Function of Frequency')
        #plt.show()
        

# Maybe don't use this method. It doesn't guarantee a lot of correlations and the correlations themselves aren't thay great. Keep it just in case we want to come
# back and use it as the main method of selecting DIBs, but for now, it is not good.
elif method.lower() == 'error':
    
    # Find the relative uncertainty for all of the DIBs. Will only use the DIBs that have an uncertainty that is no greater than
    # 5% for all sightlines. 
    
    rel_uncert = Mean_EW.div(Mean_Error)/100
    rel_uncert = rel_uncert.astype(np.float64)
    
    perc = 0.1
    
    # Create empty dataframes to hold the data for each DIB that is selected based off of the criteria 
    low_err_DIBs = pd.DataFrame(columns = Mean_EW.columns)
    bad_DIBs = pd.DataFrame()
    wavelengths = []
    
    for i in range(len(rel_uncert)):
        # check if there are any sightlines that have an error greater than 5%
        if any(rel_uncert.iloc[i] > perc):
            # do nothing 
            bad_DIBs = bad_DIBs.append(Mean_EW_Norm.iloc[i])
        else:
            wavelengths.append(DIBs['Wav'].iloc[i])
            
            if normalize.lower() == 'yes':
                low_err_DIBs = low_err_DIBs.append(Mean_EW_Norm.iloc[i])
            elif normalize.lower() == 'no':
                low_err_DIBs = low_err_DIBs.append(Mean_EW.iloc[i])
        
    # Find the pearson correlation between all good dibs
    pears_corr = low_err_DIBs.T.corr(method = 'pearson')
    fig, ax = plt.subplots(figsize = (15,10))
    sb.heatmap(pears_corr, xticklabels = wavelengths, yticklabels = wavelengths, cmap='viridis', ax = ax)
    
    

# Now find clusters of DIBs based off of their pearson correlation coefficients. Will use a heirarchical clustering algorithm.
# What it does is it takes in a similarity matrix (pears_corr), converts it to a distance matrix. The theory behind it is that 
# it takes in all of the points, treats each point as its own cluster, and then will refine it to larger clusters using an
# agglomerative approach. So it will make larger clusters by finding the distance between each, and grouping them that way,
# until a desired amount of cluster groups is formed.

# To determine a number of clusters to be formed, a dendrogram graph is formed. It shows the grouping thats made, and splits it
# up into clusters. To determine the amount of clusters to have, find the longest vertical line coming from the top, then draw a 
# straight horizontal line across it to split up the groups. Now you should have grouping based off of the colours that are 
# above and below the line.
 
# So far, this part only works if the DIB selection is selected by inputing the wavelengths of the DIBs that you want to use.
import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))
plt.title('DIB Clustering')
dend = shc.dendrogram(shc.linkage(pears_corr, method='ward'))
plt.show()

n_clusters = input('How many clusters should there be? ')

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = int(n_clusters), affinity='euclidean', linkage='ward')
groups = cluster.fit_predict(pears_corr)


group_1 = []
group_2 = []
group_3 = []
group_4 = []
group_5 = []

for k in range(len(groups)):
    if groups[k] == 0:
        group_1.append(waves[k])
    elif groups[k] == 1:
        group_2.append(waves[k])
    elif groups[k] == 2:
        group_3.append(waves[k])
    elif groups[k] == 3:
        group_4.append(waves[k])
    elif groups[k] == 4:
        group_5.append(waves[k])

print(' ')       
print('The following are clusters of DIBs \n Group 1: ', group_1, '\n Group 2: ', group_2, '\n Group 3: ', group_3, '\n Group 4: ', group_4)
