# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:07:08 2020

@author: maddy
"""
import pandas as pd

DIB_data = pd.read_excel('DIB Measurements for APO Catalog.xlsx', index_col = 0 and 1)
sight_data = pd.read_excel('Info on sight lines.xlsx', index_col = 0)
