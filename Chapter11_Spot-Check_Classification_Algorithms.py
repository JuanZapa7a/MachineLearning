# # Spot-checking is a way of discovering which algorithms perform well on your machine learning problem. You cannot know which algorithms are best suited to your problem beforehand. You must trial a number of methods and focus attention on those that prove themselves the most promising. In this chapter you will discover six machine learning algorithms that you can use when spot-checking your classification problem in Python with scikit-learn. After completing this lesson you will know:
#   1. How to spot-check machine learning algorithms on a classification problem.
#   2. How to spot-check two linear classification algorithms.
#   3. How to spot-check four nonlinear classification algorithms.

# # 1. Algorithm Spot-Checking

# You cannot know which algorithm will work best on your dataset beforehand. You must use trial and error to discover a shortlist of algorithms that do well on your problem that you can then double down on and tune further. I call this process spot-checking.

# The question is not: What algorithm should I use on my dataset? Instead it is: What algorithms should I spot-check on my dataset? You can guess at what algorithms might do well on your dataset, and this can be a good starting point. I recommend trying a mixture of algorithms and see what is good at picking out the structure in your data. Below are some suggestions when spot-checking algorithms on your dataset:
# - Try a mixture of algorithm representations (e.g. instances and trees).

# - Try a mixture of learning algorithms (e.g. different algorithms for learning the same type of representation).

# - Try a mixture of modeling types (e.g. linear and nonlinear functions or parametric and nonparametric).

# Let’s get specific. In the next section, we will look at algorithms that you can use to spot-check on your next classification machine learning project in Python.

# # 2. Algorithms Overview
# We are going to take a look at six classification algorithms that you can spot-check on your dataset. Starting with two linear machine learning algorithms:
# 􏰀 - Logistic Regression.
# 􏰀 - Linear Discriminant Analysis.
# Then looking at four nonlinear machine learning algorithms:
# 􏰀 - k-Nearest Neighbors.
# 􏰀 - Naive Bayes.
# 􏰀 - Classification and Regression Trees. 􏰀
#  - Support Vector Machines.
# Each recipe is demonstrated on the Pima Indians onset of Diabetes dataset. A test harness using 10-fold cross validation is used to demonstrate how to spot-check each machine learning algorithm and mean accuracy measures are used to indicate algorithm performance. The recipes assume that you know about each machine learning algorithm and how to use them. We will not go into the API or parameterization of each algorithm.

# # 3. Linear Machine Learning Algorithms
# This section demonstrates minimal recipes for how to use two linear machine learning algorithms: logistic regression and linear discriminant analysis.

# A. Logistic Regression
# Logistic regression assumes a Gaussian distribution for the numeric input variables and can model binary classification problems. You can construct a logistic regression model using the LogisticRegression class. http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

# %%
# Logistic Regression Classification
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:8]
y = data[:,8]
# create kfold for training and testing
kfold = KFold(n_splits=10, random_state=7, shuffle = True)
# create a model
lgr_model = LogisticRegression(max_iter=1000)
# Display accuracy
result = cross_val_score(lgr_model, X, y, cv=kfold, scoring='accuracy')
print("Logistic Regression Classification Accuracy: mean = {:.3f}%, std = {"
      ":.3f}%".format(result.mean()*100.0,result.std()*100.0))

# B. Linear Discriminant Analysis
# Linear Discriminant Analysis or LDA is a statistical technique for binary and multiclass classification. It too assumes a Gaussian distribution for the numerical input variables. You can construct a LDA model using the linerDiscriminantAnalysis class http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis. LinearDiscriminantAnalysis.html

# %%
# LDA Classification
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:8]
y = data[:,8]
# create kfold for training and testing
kfold = KFold(n_splits=10, random_state=7, shuffle = True)
# create a model
lda_model = LinearDiscriminantAnalysis()
# Display accuracy
result = cross_val_score(lda_model, X, y, cv=kfold, scoring='accuracy')
print("Linear Discriminant Analysis Accuracy: mean = {:.3f}%, std = {"
      ":.3f}%".format(result.mean()*100.0,result.std()*100.0))

# # 4. Nonlinear Machine Learning Algorithms
# This section demonstrates minimal recipes for how to use 4 nonlinear machine learning algorithms.

# A. k-Nearest Neighbors
# The k-Nearest Neighbors algorithm (or KNN) uses a distance metric to find the k most similar instances in the training data for a new instance and takes the mean outcome of the neighbors as the prediction. You can construct a KNN model using the KNeighborsClassifier class http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier. html

# %%
# KNN Classification
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:8]
y = data[:,8]
# create kfold for training and testing
kfold = KFold(n_splits=10, random_state=7, shuffle = True)
# create a model
knn_model = KNeighborsClassifier()
# Display accuracy
result = cross_val_score(knn_model, X, y, cv=kfold, scoring='accuracy')
print("k-Nearest Neighbors Accuracy: mean = {:.3f}%, std = {:.3f}%".format(
  result.mean()*100.0,result.std()*100.0))


# B. Naive Bayes
# Naive Bayes calculates the probability of each class and the conditional probability of each class given each input value. These probabilities are estimated for new data and multiplied together, assuming that they are all independent (a simple or naive assumption). When working with real-valued data, a Gaussian distribution is assumed to easily estimate the probabilities for input variables using the Gaussian Probability Density Function. You can construct a Naive Bayes model using the GaussianNB class http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html

# %%
# Gaussian Naive Bayes Classification
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:8]
y = data[:,8]
# create kfold for training and testing
kfold = KFold(n_splits=10, random_state=7, shuffle = True)
# create a model
gnb_model = GaussianNB()
# Display accuracy
result = cross_val_score(gnb_model, X, y, cv=kfold, scoring='accuracy')
print("Naive Bayes Accuracy: mean = {:.3f}%, std = {:.3f}%".format(
  result.mean()*100.0,result.std()*100.0))

# C. Classification and Regression Trees
# Classification and Regression Trees (CART or just decision trees) construct a binary tree from the training data. Split points are chosen greedily by  evaluating each attribute and each value of each attribute in the training data in order to minimize a cost function (like the Gini index). You can construct a CART model using the DecisionTreeClassifier class http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier. html

# %%
# CART Classification
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:8]
y = data[:,8]
# create kfold for training and testing
kfold = KFold(n_splits=10, random_state=7, shuffle = True)
# create a model
cart_model = DecisionTreeClassifier()
# Display accuracy
result = cross_val_score(cart_model, X, y, cv=kfold, scoring='accuracy')
print("Classification and Regression Trees Accuracy: mean = {:.3f}%, std = {"
      ":.3f}%".format(result.mean()*100.0,result.std()*100.0))

# D. Support Vector Machines
# Support Vector Machines (or SVM) seek a line that best separates two classes. Those data instances that are closest to the line that best  separates the classes are called support vectors and influence where the  line is placed. SVM has been extended to support multiple classes. Of particular importance is the use of different kernel functions via the  kernel parameter. A powerful Radial Basis Function is used by default. You can construct an SVM model using the SVC class http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

# %%
# SVM Classification
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:8]
y = data[:,8]
# create kfold for training and testing
kfold = KFold(n_splits=10, random_state=7, shuffle = True)
# create a model
svm_model = SVC()
# Display accuracy
result = cross_val_score(svm_model, X, y, cv=kfold, scoring='accuracy')
print("Support Vector Machines Accuracy: mean = {:.3f}%, std = {"
      ":.3f}%".format(result.mean()*100.0,result.std()*100.0))

