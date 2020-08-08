# # Ensembles can give you a boost in accuracy on your dataset. In this chapter you will discover how you can create some of the most powerful types of ensembles in Python using scikit-learn. This lesson will step you through Boosting, Bagging and Majority Voting and show you how you can continue to ratchet up the accuracy of the models on your own datasets. After completing this lesson you will know:
# 1. How to use bagging ensemble methods such as bagged decision trees, random forest and extra trees.
# 2. How to use boosting ensemble methods such as AdaBoost and stochastic gradient boosting.
# 3. How to use voting ensemble methods to combine the predictions from multiple algorithms.

# 1 Combine Models Into Ensemble Predictions
# The three most popular methods for combining the predictions from different models are:
# 􏰀 - Bagging. Building multiple models (typically of the same type) from different subsamples of the training dataset.
# 􏰀 - Boosting. Building multiple models (typically of the same type) each of which learns to fix the prediction errors of a prior model in the sequence of models.
# 􏰀 - Voting. Building multiple models (typically of differing types) and simple statistics (like calculating the mean) are used to combine predictions.

# This assumes you are generally familiar with machine learning algorithms and ensemble methods and will not go into the details of how the algorithms work or their parameters. The Pima Indians onset of Diabetes dataset is used to demonstrate each algorithm. Each ensemble algorithm is demonstrated using 10-fold cross validation and the classification accuracy performance metric.

# 2 Bagging Algorithms
# Bootstrap Aggregation (or Bagging) involves taking multiple samples from  your training dataset (with replacement) and training a model for each sample. The final output prediction is averaged across the predictions of all of the sub-models. The three bagging models covered in this section are as follows:
# 􏰀 - Bagged Decision Trees. 􏰀
#  - Random Forest.
# 􏰀 - Extra Trees.

# # A. Bagged Decision Trees
# Bagging performs best with algorithms that have high variance. A popular  example are decision trees, often constructed without pruning. In the example below is an example of using the BaggingClassifier with the Classification and Regression Trees algorithm (DecisionTreeClassifier). http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html. A total of 100 trees are created.

# %%
# Bagged Decision Trees for Classification
import pandas as pd
from sklearn.ensemble import BaggingClassifier
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
# evaluate models
kfold = KFold(n_splits=10, random_state=7, shuffle = True)
bag_model = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                              n_estimators=100, random_state=7)
results = cross_val_score(bag_model, X, y, cv=kfold, scoring='accuracy')
print("Accuracy: mean = {:.3f}%, std = {:.3f}%".format(results.mean()*100.0,
                                                       results.std()*100.0))

# # B. Random Forest
# Random Forests is an extension of bagged decision trees. Samples of the training dataset are taken with replacement, but the trees are constructed in a way that reduces the correlation between individual classifiers. Specifically, rather than greedily choosing the best split point in the construction of each tree, only a random subset of features are considered for each split. You can construct a Random Forest model for classification using the RandomForestClassifier 2 class http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html. The example below demonstrates using Random Forest for classification with 100 trees and split points chosen from a random selection of 3 features.

# %%
# Random Forest Classification
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:8]
y = data[:,8]
# evaluate models
kfold = KFold(n_splits=10, random_state=7, shuffle = True)
forest_model = RandomForestClassifier( max_features = 3, n_estimators=100,
                                       random_state=7)
results = cross_val_score(forest_model, X, y, cv=kfold, scoring='accuracy')
print("Accuracy: mean = {:.3f}%, std = {:.3f}%".format(results.mean()*100.0,
                                                       results.std()*100.0))

# C. Extra Trees
# Extra Trees are another modification of bagging where random trees are
# constructed from samples of the training dataset. You can construct an Extra Trees model for classification using the ExtraTreesClassifier class http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier. html. The example below provides a demonstration of extra trees with the number of trees set to 100 and splits chosen from 7 random features.

# %%
# Extra Trees Classification
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:8]
y = data[:,8]
# evaluate models
kfold = KFold(n_splits=10, random_state=7, shuffle = True)
exttrees_model = ExtraTreesClassifier( max_features = 3, n_estimators=100,
                                       random_state=7)
results = cross_val_score(exttrees_model, X, y, cv=kfold, scoring='accuracy')
print("Accuracy: mean = {:.3f}%, std = {:.3f}%".format(results.mean()*100.0,
                                                       results.std()*100.0))

# 3. Boosting Algorithms
# Boosting ensemble algorithms creates a sequence of models that attempt to correct the mistakes of the models before them in the sequence. Once created, the models make predictions which may be weighted by their demonstrated accuracy and the results are combined to create a final output prediction. The two most common boosting ensemble machine learning algorithms are:
# 􏰀 - AdaBoost.
# 􏰀 - Stochastic Gradient Boosting.

# A. AdaBoost
# AdaBoost was perhaps the first successful boosting ensemble algorithm. It generally works by weighting instances in the dataset by how easy or difficult they are to classify, allowing the algorithm to pay or less attention to them in the construction of subsequent models. You can construct an AdaBoost model for classification using the AdaBoostClassifier class http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html. The example below demonstrates the construction of 30 decision trees in sequence using the AdaBoost algorithm.

# %%
# AdaBoost Classification
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:8]
y = data[:,8]
# evaluate models
kfold = KFold(n_splits=10, random_state=7, shuffle = True)
adaboost_model = AdaBoostClassifier(n_estimators=100, random_state=7)
results = cross_val_score(adaboost_model, X, y, cv=kfold,scoring='accuracy')
print("Accuracy: mean = {:.3f}%, std = {:.3f}%".format(results.mean()*100.0,
                                                       results.std()*100.0))

# B. Stochastic Gradient Boosting
# Stochastic Gradient Boosting (also called Gradient Boosting Machines) are one of the most sophisticated ensemble techniques. It is also a technique that is proving to be perhaps one of the best techniques available for improving performance via ensembles. You can construct a Gradient Boosting model for classification using the GradientBoostingClassifier class http://scikit-learn.org/stable/modules/generated/sklearn.ensemble. GradientBoostingClassifier.html. The example below demonstrates Stochastic Gradient Boosting for classification with 100 trees.

# %%
# Stochastic Gradient Boosting Classification
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:8]
y = data[:,8]
# evaluate models
kfold = KFold(n_splits=10, random_state=7, shuffle = True)
gradboost_model = GradientBoostingClassifier(n_estimators=100, random_state=7)
results = cross_val_score(gradboost_model, X, y, cv=kfold,scoring='accuracy')
print("Accuracy: mean = {:.3f}%, std = {:.3f}%".format(results.mean()*100.0,
                                                       results.std()*100.0))

# 4. Voting Ensemble
# Voting is one of the simplest ways of combining the predictions from multiple machine learning algorithms. It works by first creating two or more standalone models from your training dataset. A Voting Classifier can then be used to wrap your models and average the predictions of the sub-models when asked to make predictions for new data. The predictions of the sub-models can be weighted, but specifying the weights for classifiers manually or even heuristically is difficult. More advanced methods can learn how to best weight the predictions from sub-models, but this is called stacking (stacked aggregation) and is currently not provided in scikit-learn.

# You can create a voting ensemble model for classification using the VotingClassifier http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html class . The code below provides an example of combining the predictions of logistic regression, classification and regression trees and support vector machines together for a classification problem.

# %%
# Voting Ensemble for Classification
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:8]
y = data[:,8]
# evaluate models
kfold = KFold(n_splits=10, random_state=7, shuffle = True)
# create the sub models
models = []
lgr_model = LogisticRegression(max_iter = 1000)
models.append(('logistic', lgr_model))
cart_model = DecisionTreeClassifier()
models.append(('cart', cart_model))
svm_model = SVC()
models.append(('svm', svm_model))
# create the ensemble model
ensemble = VotingClassifier(models)
results = cross_val_score(ensemble, X, y, cv=kfold, scoring='accuracy')
print("Accuracy: mean = {:.3f}%, std = {:.3f}%".format(results.mean()*100.0,
                                                       results.std()*100.0))