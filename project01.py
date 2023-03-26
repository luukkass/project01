#imports
import pandas as pd

'''Requirements on preprocessing
Any two of the following operations are mandatory:
remove rows based on subsetting
derive new columns
use aggregation operators
treat missing values'''

data = pd.read_csv("speeddating.csv")
print(data.head())

'''Requirements on model building
Use any classifier. Choose one of the following two options:

perform train/test split
use crossvalidation
Also, evaluate and compare at least two algorithms of different types (e.g. logistic regression and random forest).

Python: use any classifier from sklearn'''


'''Requirements on metaparameter tuning
If the chosen classifier has any metaparameters that can be tuned, use one of the following methods:

try several configurations and describe the best result in the final report
perform grid search or other similar automatic method
once you have tuned metaparameters on a dedicated development (training) set, e.g. with GridSearchCV, you can retrain the model on the complete training data, as e.g. described here for Python: https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html and https://stackoverflow.com/questions/26962050/confused-with-repect-to-working-of-gridsearchcv
Python recommendation: sklearn.model_selection.GridSearchCV'''
