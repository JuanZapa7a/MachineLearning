# You need to see how all of the pieces of a predictive modeling machine learning project actually fit together. In this lesson you will complete your first machine learning project using Python. In this step-by-step tutorial project you will:
# 􏰀- Download and install Python SciPy and get the most useful package for
# machine learning in Python.
# 􏰀- Load a dataset and understand it’s structure using statistical summaries
# and data visual- ization.
# 􏰀- Create 6 machine learning models, pick the best and build confidence
# that the accuracy is reliable.
# If you are a machine learning beginner and looking to finally get started using Python, this tutorial was designed for you.

# 1 The Hello World of Machine Learning
# The best small project to start with on a new tool is the classification of iris flowers. This is a good dataset for your first project because it is so well understood.
# 􏰀
# - Attributes are numeric so you have to figure out how to load and handle data.
# - It is a classification problem, allowing you to practice with an easier type of supervised learning algorithm.
# - It is a multiclass classification problem (multi-nominal) that may require some specialized handling.
# - It only has 4 attributes and 150 rows, meaning it is small and easily fits into memory (and a screen or single sheet of paper).
# - All of the numeric attributes are in the same units and the same scale not requiring any special scaling or transforms to get started.

# In this tutorial we are going to work through a small machine learning  project end-to-end. Here is an overview of what we are going to cover:
# 1. Loading the dataset.
# 2. Summarizing the dataset.
# 3. Visualizing the dataset.
# 4. Evaluating some algorithms.
# 5. Making some predictions.

# Take your time and work through each step. Try to type in the commands yourself or copy-and-paste the commands to speed things up. Start your Python interactive environment and let’s get started with your hello world machine learning project in Python.

# %%
####################
# 1. Load libraries
####################
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# The iris dataset can be downloaded from the UCI Machine Learning repository https://goo.gl/mLmoIz. We are using Pandas to load the data. We will also use Pandas next to explore the data both with descriptive statistics and data visualization. Note that we are specifying the names of each column when loading the data. This will help later when we explore the data.

#################
# 2. Load dataset
#################
filename = 'iris.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataframe = pd.read_csv(filename, names=names)

##########################
# 3 Summarize the Dataset
##########################
# Now it is time to take a look at the data. In this step weare going to take a look at the data a few different ways:

# 􏰀- Dimensions of the dataset.
# shape
print(dataframe.shape)

# 􏰀- Peek at the data itself.
# head (+ 20 lines).
print(dataframe.head(20))

# 􏰀- Statistical summary of all attributes.
# Now we can take a look at a summary of each attribute. This includes the  count, mean, the min and max values as well as some percentiles descriptions
print(dataframe.describe())

# - Class Distribution
# Let’s now take a look at the number of instances (rows) that belong to each class. We can view this as an absolute count.
# class distribution
print(dataframe.groupby('class').size())

########################
# 4. Data Visualization
########################
# We now have a basic idea about the data. We need to extend this with some visualizations. We are going to look at two types of plots:
# 􏰀 - Univariate plots to better understand each attribute.
# 􏰀 - Multivariate plots to better understand the relationships between attributes.

# Univariate plots: box and whisker plots
dataframe.plot(kind='box', subplots=True, layout=(2,2), sharex=False,
            sharey=False)
plt.show()

# Univariate plots: histograms
dataframe.hist()
plt.show()

# Multivariate plots: scatter plot matrix
pd.plotting.scatter_matrix(dataframe)
plt.show()

##############################
# 5. Evaluate Some Algorithms
##############################
# Now it is time to create some models of the data and estimate their accuracy on unseen data. Here is what we are going to cover in this step:
# 1. Separate out a validation dataset.
# Split-out validation dataset
data = dataframe.values
X = data[:,0:4]
y = data[:,4]
# create split train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    random_state=7)
print("Size of training set = train {}, labels {}".format(X_train.shape,
                                                          y_train.shape))
print("Size of test set = test {}, labels {}".format(X_test.shape,
                                                     y_test.shape))
# 2. Setup the test harness to use 10-fold cross validation.
# We will use 10-fold cross validation to estimate accuracy. This will split our dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits. We are using the metric of accuracy to evaluate models. This is a ratio of the number of correctly predicted instances divided by the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate). We will be using the scoring variable when we run build and evaluate each model next.
# create kfold for training and testing
kfold = KFold(n_splits=10, random_state=7, shuffle = True)

# 3. Build 5 different models to predict species from flower measurements  prepare models
# We don’t know which algorithms would be good on this problem or what configurations to use. We get an idea from the plots that some of the classes are partially linearly separable in some dimensions, so we are expecting generally good results. Let’s evaluate six different algorithms:
# - Logistic Regression (LR).
# - Linear Discriminant Analysis (LDA).
# 􏰀- k-Nearest Neighbors (KNN).
# 􏰀- Classification and Regression Trees (CART). 􏰀
# - Gaussian Naive Bayes (NB).
# - Support Vector Machines (SVM).

# This list is a good mixture of simple linear (LR and LDA), nonlinear (KNN,
# CART, NB and SVM) algorithms. We reset the random number seed before each
# run to ensure that the evaluation of each algorithm is performed using
# exactly the same data splits. It ensures the results are directly comparable. Let’s build and evaluate our six models:

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
print("We now have 6 models and accuracy estimations for each.\nWe need to "
      "compare the models to each other and select the most accurate.\n"
      "Running the example above, we get the following raw results:")
for name, model in models:
  # kfold = KFold(n_splits=10, random_state=7, shuffle = True)
  cv_results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
  results.append(cv_results)
  names.append(name)
  print("{} Accuracy: mean = {:.3f}%, std = {:.3f}%".format(name,
                                                            cv_results.mean()*100.0,
                                                            cv_results.std()*100.0))

# 4. Select the best model.
print("We can see that it looks like KNN has the largest estimated accuracy "
      "score.\nWe can also create a plot of the model evaluation results and compare the spread and the mean accuracy of each model.\nThere is a population of accuracy measures for each algorithm because each algorithm was evaluated 10 times (10 fold cross validation).")
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

print("You can see that the box and whisker plots are squashed at the top of the range, with many samples achieving 100% accuracy.")

# 5. Make Predictions
# The KNN algorithm was the most accurate model that we tested. Now we want to get an idea of the accuracy of the model on our validation dataset. This will give us an independent final check on the accuracy of the best model. It is important to keep a validation set just in case you made a slip during training, such as overfitting to the training set or a data leak. Both will result in an overly optimistic result. We can run the KNN model directly on the validation set and summarize the results as a final accuracy score, a confusion matrix and a classification report.

# Make predictions on validation dataset
print("The KNN algorithm was the most accurate model that we tested. Now we want to get an idea of the accuracy of the model on our validation dataset. This will give us an independent final check on the accuracy of the best model. It is important to keep a validation set just in case you made a slip during training, such as overfitting to the training set or a data leak. Both will result in an overly optimistic result. We can run the KNN model directly on the validation set and summarize the results as a final accuracy score, a confusion matrix and a classification report.")
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
