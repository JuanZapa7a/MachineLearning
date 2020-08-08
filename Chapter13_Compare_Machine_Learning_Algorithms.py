# # It is important to compare the performance of multiple different machine learning algorithms consistently. In this chapter you will discover how you can create a test harness to compare multiple different machine learning algorithms in Python with scikit-learn. You can use this test harness as a template on your own machine learning problems and add more and different algorithms to compare. After completing this lesson you will know:

# 1. How to formulate an experiment to directly compare machine learning algorithms.
# 2. A reusable template for evaluating the performance of multiple algorithms on one dataset.
# 3. How to report and visualize the results when comparing algorithm performance.

# 1. Choose The Best Machine Learning Model
# When you work on a machine learning project, you often end up with multiple good models to choose from. Each model will have different performance characteristics. Using resampling methods like cross validation, you can get an estimate for how accurate each model may be on unseen data. You need to be able to use these estimates to choose one or two best models from the suite of models that you have created.

# When you have a new dataset, it is a good idea to visualize the data using different techniques in order to look at the data from different perspectives. The same idea applies to model selection. You should use a number of different ways of looking at the estimated accuracy of your machine learning algorithms in order to choose the one or two algorithm to finalize. A way to do this is to use visualization methods to show the average accuracy, variance and other properties of the distribution of model accuracies. In the next section you will discover exactly how you can do that in Python with scikit-learn.

# 2. Compare Machine Learning Algorithms Consistently
# The key to a fair comparison of machine learning algorithms is ensuring that each algorithm is evaluated in the same way on the same data. You can achieve this by forcing each algorithm to be evaluated on a consistent test harness. In the example below six different classification algorithms are compared on a single dataset:
# 􏰀 - Logistic Regression.
# 􏰀 - Linear Discriminant Analysis.
# 􏰀 - k-Nearest Neighbors.
# 􏰀 - Classification and Regression Trees. 􏰀 Naive Bayes.
# 􏰀 - Support Vector Machines.

# The dataset is the Pima Indians onset of diabetes problem. The problem has two classes and eight numeric input variables of varying scales. The 10-fold cross validation procedure is used to evaluate each algorithm, importantly configured with the same random seed to ensure that the same splits to the training data are performed and that each algorithm is evaluated in precisely the same way. Each algorithm is given a short name, useful for summarizing results afterward.

# %%
# Compare Algorithms
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:8]
y = data[:,8]
# prepare models
models = []
models.append(['LR', LogisticRegression(max_iter = 1000)])
models.append(['LDA', LinearDiscriminantAnalysis()])
models.append(['KNN', KNeighborsClassifier()])
models.append(['CART', DecisionTreeClassifier()])
models.append(['NB', GaussianNB()])
models.append(['SVM', SVC()])
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
  kfold = KFold(n_splits=10, random_state=7, shuffle = True)
  cv_results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
  results.append(cv_results)
  names.append(name)
  print("{} Accuracy: mean = {:.3f}%, std = {"
        ":.3f}%".format(name ,cv_results.mean()*100.0,cv_results.std()*100.0))
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# From these results, it would suggest that both logistic regression and
# linear discriminate analysis are perhaps worthy of further study on this problem.º