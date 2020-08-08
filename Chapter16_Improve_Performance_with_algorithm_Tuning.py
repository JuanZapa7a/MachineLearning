# # Machine learning models are parameterized so that their behavior can be tuned for a given problem. Models can have many parameters and finding the best combination of parameters can be treated as a search problem. In this chapter you will discover how to tune the parameters of machine learning algorithms in Python using the scikit-learn. After completing this lesson you will know:
#   1. The importance of algorithm parameter tuning to improve algorithm performance.
#   2. How to use a grid search algorithm tuning strategy.
#   3. How to use a random search algorithm tuning strategy.

# 1  Machine Learning Algorithm Parameters
# Algorithm tuning is a final step in the process of applied machine learning before finalizing your model. It is sometimes called hyperparameter optimization where the algorithm parameters are referred to as hyperparameters, whereas the coefficients found by the machine learning algorithm itself are referred to as parameters. Optimization suggests the search-nature of the problem. Phrased as a search problem, you can use different search strategies to find a good and robust parameter or set of parameters for an algorithm on a given problem. Python scikit-learn provides two simple methods for algorithm parameter tuning:
# 􏰀 Grid Search Parameter Tuning.
# 􏰀 Random Search Parameter Tuning.

# 2 Grid Search Parameter Tuning
# Grid search is an approach to parameter tuning that will methodically build and evaluate a model for each combination of algorithm parameters specified in a grid. You can perform a grid search using the GridSearchCV class http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html. The example below evaluates different alpha values for the Ridge Regression algorithm on the standard diabetes dataset. This is a one-dimensional grid search.

# %%
# Grid Search for Algorithm Tuning
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:8]
y = data[:,8]
# evaluate models in a grid
ticks = np.array([1,0.1,0.01,0.001,0.0001,0])
param_grid = dict(alpha=ticks)
ridge_model = Ridge()
grid = GridSearchCV(estimator=ridge_model, param_grid=param_grid)
grid.fit(X, y)
print("Best Score:{}".format(grid.best_score_))
print("Best Parameter:{}".format(grid.best_estimator_.alpha))

# 3 Random Search Parameter Tuning
# Random search is an approach to parameter tuning that will sample algorithm parameters from a random distribution (i.e. uniform) for a fixed number of  iterations. A model is constructed and evaluated for each combination of parameters chosen. You can perform a random search    for algorithm parameters using the RandomizedSearchCV class http://scikit-learn.org/stable/modules/generated/sklearn.model_selection. RandomizedSearchCV.html. The example below evaluates different random alpha values between 0 and 1 for the Ridge Regression algorithm on the standard diabetes dataset. A total of 100 iterations are performed with uniformly random alpha values selected in the range between 0 and 1 (the range that alpha values can take).

# %%
import pandas as pd
from scipy.stats import uniform
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:8]
y = data[:,8]
# evaluate models in a grid
param_distributions = dict(alpha=uniform())
ridge_model = Ridge()
rand_search = RandomizedSearchCV(estimator=ridge_model,
                          param_distributions=param_distributions)
rand_search.fit(X, y)
print("Best Score:{}".format(rand_search.best_score_))
print("Best Parameter:{}".format(rand_search.best_estimator_.alpha))