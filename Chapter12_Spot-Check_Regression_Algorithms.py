# # Spot-checking is a way of discovering which algorithms perform well on your machine learning problem. You cannot know which algorithms are best suited to your problem beforehand. You must trial a number of methods and focus attention on those that prove themselves the most promising. In this chapter you will discover six machine learning algorithms that you can use when spot-checking your classification problem in Python with scikit-learn. After completing this lesson you will know:
#   1. How to spot-check machine learning algorithms on a regression problem.
#   2. How to spot-check two linear regression algorithms.
#   3. How to spot-check four nonlinear regression algorithms.

# # 1. Algorithms Overview
# In this lesson we are going to take a look at seven regression algorithms that you can spot-check on your dataset. Starting with four linear machine learning algorithms:
# 􏰀 - Linear Regression.
# 􏰀 - Ridge Regression.
# 􏰀 - LASSO Linear Regression. 􏰀
#  - Elastic Net Regression.
# Then looking at three nonlinear machine learning algorithms:
# 􏰀 - k-Nearest Neighbors.
# 􏰀 - Classification and Regression Trees. 􏰀
#  - Support Vector Machines.

# Each recipe is demonstrated on the Boston House Price dataset. This is a regression problem where all attributes are numeric. A test harness with 10-fold cross validation is used to demonstrate how to spot-check each machine learning algorithm and mean squared error measures are used to indicate algorithm performance. Note that mean squared error values are inverted (negative). This is a quirk of the cross val score() function used that requires all algorithm metrics to be sorted in ascending order (larger value is better). The recipes assume that you know about each machine learning algorithm and how to use them. We will not go into the API or parameterization of each algorithm.

# # 2. Linear Machine Learning Algorithms
# This section provides examples of how to use four different linear machine learning algorithms for regression in Python with scikit-learn.

# A. Linear Regression
# Linear regression assumes that the input variables have a Gaussian distribution. It is also assumed that input variables are relevant to the output variable and that they are not highly correlated with each other (a problem called collinearity). You can construct a linear regression model using the LinearRegression class. http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression. html

# %%
# Linear Regression
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# load data
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
         'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pd.read_csv(filename, delim_whitespace=True, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:13]
y = data[:,13]
# create kfold for training and testing
kfold = KFold(n_splits=10, random_state=7, shuffle = True)
# create a model
lnr_model = LinearRegression()
# Display accuracy
result = cross_val_score(lnr_model, X, y, cv=kfold,
                         scoring='neg_mean_absolute_error')
print("Linear Regression with scoring Mean Absolute Error: mean = {:.3f}, "
      "std = {:.3f}".format(result.mean(), result.std()))
# A value of 0 indicates no error or perfect predictions. Like logloss, this metric is inverted by the cross val score() function.

# B. Ridge Regression
# Ridge regression is an extension of linear regression where the loss function is modified to minimize the complexity of the model measured as the sum squared value of the coefficient values (also called the L2-norm). You can construct a ridge regression  model by using the Ridge class

# %%
# Ridge Regression
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

# load data
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
         'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pd.read_csv(filename, delim_whitespace=True, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:13]
y = data[:,13]
# create kfold for training and testing
kfold = KFold(n_splits=10, random_state=7, shuffle = True)
# create a model
ridge_model = Ridge()
# Display accuracy
result = cross_val_score(ridge_model, X, y, cv=kfold,
                         scoring='neg_mean_absolute_error')
print("Ridge Regression with scoring Mean Absolute Error: mean = {:.3f}, "
      "std = {:.3f}".format(result.mean(), result.std()))

# C. LASSO Regression
# The Least Absolute Shrinkage and Selection Operator (or LASSO for short) is a modification of linear regression, like ridge regression, where the loss function is modified to minimize the complexity of the model measured as the sum absolute value of the coefficient values (also called the L1-norm). You can construct a LASSO model by using the Lasso class.

# %%
# Lasso Regression
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso

# load data
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
         'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pd.read_csv(filename, delim_whitespace=True, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:13]
y = data[:,13]
# create kfold for training and testing
kfold = KFold(n_splits=10, random_state=7, shuffle = True)
# create a model
lasso_model = Lasso()
# Display accuracy
result = cross_val_score(ridge_model, X, y, cv=kfold,
                         scoring='neg_mean_absolute_error')
print("Lasso Regression with scoring Mean Absolute Error: mean = {:.3f}, "
      "std = {:.3f}".format(result.mean(), result.std()))

# D. ElasticNet Regression
# ElasticNet is a form of regularization regression that combines the properties of both Ridge Regression and LASSO regression. It seeks to minimize the complexity of the regression model (magnitude and number of regression coefficients) by penalizing the model using both the L2-norm (sum squared coefficient values) and the L1-norm (sum absolute coefficient values). You can construct an ElasticNet model using the ElasticNet class. http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html

# %%
# ElasticNet Regression
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet

# load data
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
         'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pd.read_csv(filename, delim_whitespace=True, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:13]
y = data[:,13]
# create kfold for training and testing
kfold = KFold(n_splits=10, random_state=7, shuffle = True)
# create a model
enet_model = ElasticNet()
# Display accuracy
result = cross_val_score(enet_model, X, y, cv=kfold,
                         scoring='neg_mean_absolute_error')
print("ElasticNet Regression with scoring Mean Absolute Error: mean = {:.3f}, "
      "std = {:.3f}".format(result.mean(), result.std()))

# # 3. Nonlinear Machine Learning Algorithms
# This section provides examples of how to use three different nonlinear machine learning algorithms for regression in Python with scikit-learn.

# A. K-Nearest Neighbors
# The k-Nearest Neighbors algorithm (or KNN) locates the k most similar instances in the training dataset for a new data instance. From the k neighbors, a mean or median output variable is taken as the prediction. Of note is the distance metric used (the metric argument). The Minkowski distance is used by default, which is a generalization of both the Euclidean distance (used when all inputs have the same scale) and Manhattan distance (for when the scales of the input variables differ). You can construct a KNN model for regression using the KNeighborsRegressor class. http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor. html

# %%
# KNN Regression
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor

# load data
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
         'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pd.read_csv(filename, delim_whitespace=True, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:13]
y = data[:,13]
# create kfold for training and testing
kfold = KFold(n_splits=10, random_state=7, shuffle = True)
# create a model
knn_model = KNeighborsRegressor()
# Display accuracy
result = cross_val_score(knn_model, X, y, cv=kfold,
                         scoring='neg_mean_absolute_error')
print("KNN Regression with scoring Mean Absolute Error: mean = {:.3f}, std = {:.3f}".format(result.mean(), result.std()))

# B. Classification and Regression Trees
# Decision trees or the Classification and Regression Trees (CART as they are
# known) use the training data to select the best points to split the data in
# order to minimize a cost metric. The default cost metric for regression
# decision trees is the mean squared error, specified in the criterion
# parameter. You can create a CART model for regression using the
# DecisionTreeRegressor class. http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

# %%
# Decision Tree Regression
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

# load data
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
         'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pd.read_csv(filename, delim_whitespace=True, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:13]
y = data[:,13]
# create kfold for training and testing
kfold = KFold(n_splits=10, random_state=7, shuffle = True)
# create a model
cart_model = DecisionTreeRegressor()
# Display accuracy
result = cross_val_score(cart_model, X, y, cv=kfold,
                         scoring='neg_mean_absolute_error')
print("Classification and Regression Trees Regression with scoring Mean Absolute Error: mean = {:.3f}, std = {:.3f}".format(result.mean(), result.std()))

# C. Support Vector Machines
# Support Vector Machines (SVM) were developed for binary classification. The technique has been extended for the prediction real-valued problems called Support Vector Regression (SVR). Like the classification example, SVR is built upon the LIBSVM library. You can create an SVM model for regression using the SVR class. http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

# %%
# SVM Regression
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR

# load data
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
         'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pd.read_csv(filename, delim_whitespace=True, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:13]
y = data[:,13]
# create kfold for training and testing
kfold = KFold(n_splits=10, random_state=7, shuffle = True)
# create a model
svm_model = SVR()
# Display accuracy
result = cross_val_score(svm_model, X, y, cv=kfold,
                         scoring='neg_mean_absolute_error')
print("Support Vector Machines Regression with scoring Mean Absolute Error: mean = {:.3f}, std = {:.3f}".format(result.mean(), result.std()))

