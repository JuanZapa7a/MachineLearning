# # There are standard workflows in a machine learning project that can be automated. In Python scikit-learn, Pipelines help to clearly define and automate these workflows. In this chapter you will discover Pipelines in scikit-learn and how you can automate common machine learning workflows. After completing this lesson you will know:
# 1. How to use pipelines to minimize data leakage.
# 2. How to construct a data preparation and modeling pipeline.
# 3. How to construct a feature extraction and modeling pipeline.

# 1. Automating Machine Learning Workflows
# There are standard workflows in applied machine learning. Standard because they overcome common problems like data leakage in your test harness. Python scikit-learn provides a Pipeline utility to help automate machine learning workflows. Pipelines work by allowing for a linear sequence of data transforms to be chained together culminating in a modeling process that can be evaluated.
# The goal is to ensure that all of the steps in the pipeline are constrained to the data available for the evaluation, such as the training dataset or each fold of the cross validation procedure. You can learn more about Pipelines in scikit-learn by reading the Pipeline section of the user guide. You can also review the API documentation for the Pipeline and FeatureUnion classes and the pipeline module.
# http://scikit-learn.org/stable/modules/pipeline.html
# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline

# 2. Data Preparation and Modeling Pipeline
# An easy trap to fall into in applied machine learning is leaking data from your training dataset to your test dataset. To avoid this trap you need a robust test harness with strong separation of training and testing. This includes data preparation. Data preparation is one easy way to leak knowledge of the whole training dataset to the algorithm. For example, preparing your data using normalization or standardization on the entire training dataset before learning would not be a valid test because the training dataset would have been influenced by the scale of the data in the test set.
# Pipelines help you prevent data leakage in your test harness by ensuring that data preparation like standardization is constrained to each fold of your cross validation procedure. The example below demonstrates this important data preparation and model evaluation workflow on the Pima Indians onset of diabetes dataset. The pipeline is defined with two steps:
# 1. Standardize the data.
# 2. Learn a Linear Discriminant Analysis model.

# %%
# Create a pipeline that standardizes the data then creates a model
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:8]
y = data[:,8]
# create pipeline
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('lda', LinearDiscriminantAnalysis()))
model = Pipeline(estimators)
# evaluate pipeline
kfold = KFold(n_splits=10, random_state=7, shuffle = True)
results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
print("LDA Accuracy: mean = {:.3f}%, std = {:.3f}%".format(results.mean()*100.0, results.std()*100.0))

# Notice how we create a Python list of steps that are provided to the Pipeline for process the data. Also notice how the Pipeline itself is treated like an estimator and is evaluated in its entirety by the k-fold cross validation procedure. Running the example provides a summary of accuracy of the setup on the dataset.

# 3. Feature Extraction and Modeling Pipeline
# Feature extraction is another procedure that is susceptible to data leakage. Like data preparation, feature extraction procedures must be restricted to the data in your training dataset. The pipeline provides a handy tool called the FeatureUnion which allows the results of multiple feature selection and extraction procedures to be combined into a larger dataset on which a model can be trained. Importantly, all the feature extraction and the feature union occurs within each fold of the cross validation procedure. The example below demonstrates the pipeline defined with four steps:
#   1. Feature Extraction with Principal Component Analysis (3 features).
#   2. Feature Extraction with Statistical Selection (6 features).
#   3. Feature Union.
#   4. Learn a Logistic Regression Model.

# %%
# Create a pipeline that extracts features from the data then creates a model
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
# create X and y (display shapes)
data = dataframe.values
X = data[:,0:8]
y = data[:,8]
# create feature union
features = []
features.append(('pca', PCA(n_components=3)))
features.append(('select_best', SelectKBest(k=6)))
feature_union = FeatureUnion(features)
# create pipeline
estimators = []
estimators.append(('feature_union', feature_union))
estimators.append(('logistic', LogisticRegression(max_iter = 1000)))
model = Pipeline(estimators)
# evaluate pipeline
kfold = KFold(n_splits=10, random_state=7, shuffle = True)
results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
print("Accuracy: mean = {:.3f}%, std = {:.3f}%".format(results.mean()*100.0,
                                                       results.std()*100.0))
