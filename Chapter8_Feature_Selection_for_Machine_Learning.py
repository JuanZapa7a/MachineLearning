# # The data features that you use to train your machine learning models have
# a huge influence on the performance you can achieve. Irrelevant or partially
# relevant features can negatively impact model performance. In this chapter you
# will discover automatic feature selection techniques that you can use to
# prepare your machine learning data in Python with scikit-learn. After
# completing this lesson you will know how to use:
#   1. Univariate Selection.
#   2. Recursive Feature Elimination.
#   3. Principle Component Analysis.
#   4. Feature Importance.

# # 1. Feature Selection
# Feature selection is a process where you automatically select those
# features in your data that contribute most to the prediction variable or
# output in which you are interested. Having irrelevant features in your data
# can decrease the accuracy of many models, especially linear algorithms like
# linear and logistic regression. Three benefits of performing feature
# selection before modeling your data are:
#  - Reduces Overfitting: Less redundant data means less opportunity to make
#   decisions based on noise.
#  - Improves Accuracy: Less misleading data means modeling accuracy improves.
#  - Reduces Training Time: Less data means that algorithms train faster.

# You can learn more about feature selection with scikit-learn in the article
# Feature selection.
# http://scikit-learn.org/stable/modules/feature_selection.html
# Each feature selection recipes will use the Pima Indians onset of diabetes
# dataset.

# # 2. Univariate Selection
# Statistical tests can be used to select those features that have the
# strongest relationship with the output variable. The scikit-learn library
# provides the SelectKBest class2 that can be used with a suite of different
# statistical tests to select a specific number of features. The example below
# uses the chi-squared (chi2) statistical test for non-negative features to
# select 4 of the best features from the Pima Indians onset of diabetes dataset.

import numpy as np
# %%
# Feature Extraction with Univariate Statistical Tests (Chi-squared for
# classification)
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
data = dataframe.values
# separate array data into input and output components
X = data[:,0:8] # data
Y = data[:,8] # classes or labels

# feature extraction
kbest = SelectKBest(score_func=chi2, k=4) # select the best 4 features
kbest_X = kbest.fit(X, Y) # obtain relation between X and Y
features = kbest_X.transform(X) # or features = best.fit_transform(X,Y)

# summarize scores
np.set_printoptions(precision=3)
print("Scores for each attribute: {}".format(kbest_X.scores_))

# summarize selected features
np.set_printoptions(precision=3)
print("Best 4 attributes with highest scores: {}".format(features[0:5,:])) #

# How can you obtain the names of the 4 highest scores?
idx = np.argpartition(kbest_X.scores_, -4)[-4:] # choose last 4 higher(unsorted)
indices = idx[np.argsort((-kbest_X.scores_)[idx])] # 4 higher sorted
kbestnames = [names[i] for i in indices] # iteratively fill 4 bestnames
print("4 attributes chosen with the highest scores are: {}".format(kbestnames))

# # 3. Recursive Feature Elimination
# The Recursive Feature Elimination (or RFE) works by recursively removing
# attributes and building a model on those attributes that remain. It uses the
# model accuracy to identify which attributes (and combination of attributes)
# contribute the most to predicting the target attribute. You can learn more
# about the RFE class in the scikit-learn documentation.
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html# sklearn.feature_selection.RFE
# The example below uses RFE with the logistic regression algorithm to select
# the top 3 features. The choice of algorithm does not matter too much as
# long as it is skillful and consistent.

# %%
# Feature Extraction with RFE
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
data = dataframe.values
# separate array data into input and output components
X = data[:,0:8] # data
Y = data[:,8] # classes or labels

# feature extraction
lgr_model = LogisticRegression(max_iter=1000)
rfe = RFE(lgr_model, 3) # select the best 3 features for this estimator
rfe_X = rfe.fit(X, Y) # obtain relation between X and Y
# features = rfe_X.transform(X) # or features = best.fit_transform(X,Y)

# summarize features
np.set_printoptions(precision=3)
print("Number of features: {}".format(rfe_X.n_features_))
print("Number of features to select (true) in the array: {}".format(
  rfe_X.support_))
print("Feature Ranking (1) in the array: {}".format(rfe_X.ranking_))

idx = np.argpartition(rfe_X.support_, -3)[-3:] # choose last 3 higher(unsorted)
indices = idx[np.argsort((rfe_X.support_)[idx])] # 3 higher sorted
rfenames = [names[i] for i in indices] # iteratively fill 4 bestnames
print("3 attributes chosen with the highest scores are: {}".format(rfenames))

# # 4. Principal Component Analysis
# Principal Component Analysis (or PCA) uses linear algebra to transform the
# dataset into a compressed form. Generally this is called a data reduction
# technique. A property of PCA is that you can choose the number of
# dimensions or principal components in the transformed result. In the
# example below, we use PCA and select 3 principal components. Learn more about
# the PCA 4 class in scikit-learn by reviewing the API.
# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

# %%
# Feature Extraction with PCA
import pandas as pd
from sklearn.decomposition import PCA

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
data = dataframe.values
# separate array data into input and output components
X = data[:,0:8] # data
Y = data[:,8] # classes or labels

# feature extraction
pca = PCA(n_components=3)
pca_X = pca.fit(X)
# summarize components
print("Explained Variance: {}".format(pca_X.explained_variance_ratio_))
print("Singular Values: {}".format(pca_X.singular_values_))
print(pca_X.components_)

# # 5. Feature Importance
# Bagged decision trees like Random Forest and Extra Trees can be used to
# estimate the importance of features. In the example below we construct a
# ExtraTreesClassifier classifier for the Pima Indians onset of diabetes
# dataset. You can learn more about the ExtraTreesClassifier class in the
# scikit-learn API.
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html

# %%
# Feature Importance with Extra Trees Classifier
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
data = dataframe.values
# separate array data into input and output components
X = data[:,0:8] # data
Y = data[:,8] # classes or labels

# feature extraction
etc = ExtraTreesClassifier()
etc_X = etc.fit(X, Y)
# summarize components
print("Features importance: {}".format(etc_X.feature_importances_))

idx = np.argpartition(etc_X.feature_importances_, -3)[-3:] # choose last 3
# higher(unsorted)
indices = idx[np.argsort((etc_X.feature_importances_)[idx])] # 3 higher sorted
etcnames = [names[i] for i in indices] # iteratively fill 4 bestnames
print("3 attributes chosen with the highest scores are: {}".format(etcnames))
