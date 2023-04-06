"""imports and randomstate
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
coding int values for strings"""

# Import preliminary modules
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np
from scipy.stats import norm, stats

import warnings
warnings.filterwarnings('ignore')
random_seed = 123

# For reproducibility
np.random.seed(random_seed)

# Load data
df = pd.read_csv("speeddating.csv")
print('Dataset size: ', df.shape)
print(df.info())

# Check data stats for EDA
df.describe()

# Check for missing values
df.isnull().any().any()
df.nunique().sort_values()

### Data Preprocessing

# Remove unnecessary data 
df[df=='?'] = np.nan
df = df.dropna()

# Delete till columns have only one unique value
drop_column = df.nunique().sort_values().reset_index().rename(columns = {'index':'column'})
drop = [drop_column.column[i]
        for i in range(len(drop_column)-1)
        if drop_column[0][i]<2]
df = df.drop(columns=drop, axis=1)
print(df.head())

# Convert categorical and textual data to numbers
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        oe.fit(df[[col]])
        df[col] = oe.fit_transform(df[[col]])
        
df.dropna(inplace=True)
print(df.head())

# Initialize data
X = df.drop(['match'], axis=1)
y = df['match']
print(y)

### Model Creation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Split dataset into training and test
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.25, random_state=random_seed)

# Instantiate classifier
logreg = LogisticRegression()

print(X_train.head(), y_train.head())
# Fit logreg into the training data
logreg.fit(X_train, y_train)

# Predict on the test data
y_pred = logreg.predict(X_test)

### Hyperparameter Tuning
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

tree = DecisionTreeClassifier()
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)
tree_cv.fit(X, y)
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))