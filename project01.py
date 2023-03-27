#imports and randomstate
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("speeddating.csv")
#indexing column names to delete trash ones 
colss = list(map(list, zip(df.columns.values, range(123))))
print(colss)
columnstodelete = [0,6,12,13,21,22,23,24,25,26,33,34,35,36,37,38,45,46,47,48,49,50,56,57,58,59,60,67,68,69,70,71,72,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,108,112,113,114,117,118]
#dropping
df2 = df.drop(df.columns[columnstodelete],axis = 1)
#coding int values for strings 

