# %% [markdown]
# # Introduction TODO
# Stroke is one of the leading causes of death in the United States. Every year 
# nearly 800,000 Americans suffer from stroke and  

# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn import tree

# %%
# read in data
raw_data = pd.read_csv("healthcare-dataset-stroke-data.csv")
raw_data = raw_data.drop("id", axis=1)
raw_data.shape

# %% [markdown]
# ### Data Cleaning and Preparation

# %%
raw_data.head(10)

# %% [markdown]
# Before we are able to model stroke, we have to 

# %%

# TODO Display the percentage of data that has "Unknown" smoking_status
# TODO 
# TODO Display the percentage of the data with NaN bmi



# %% [markdown]
# We begin by looking at the number of missing values there are in the dataset.

# %%
raw_data.isnull().sum()

# %% [markdown]
# Let us observe the distribution of the data before and after we drop the null values in the BMI column.

# %%
dropped_data = raw_data.dropna(axis=0, subset="bmi") # TODO 

# %%
# distribution of stroke label before and after
raw_stroke_counts = raw_data.loc[:, "stroke"].value_counts()
dropped_stroke_counts = dropped_data.loc[:, "stroke"].value_counts()

print(raw_stroke_counts[0] / raw_stroke_counts[1])
print(dropped_stroke_counts[0] / dropped_stroke_counts[1])

# %%
raw_hyper_counts = raw_data.loc[:, "hypertension"].value_counts()
dropped_hyper_counts = dropped_data.loc[:, "hypertension"].value_counts()

print(raw_hyper_counts[0] / raw_hyper_counts[1])
print(dropped_hyper_counts[0] / dropped_hyper_counts[1])


# %%
raw_data.loc[:, "smoking_status"].value_counts().plot(kind="bar", title="Distribution of Smoking Status Before Dropped Data")

# %%
dropped_data.loc[:, "smoking_status"].value_counts().plot(kind="bar", title="Distribution of Smoking Status After Dropped Data")

# %%
binary_data_x = pd.get_dummies(dropped_data.iloc[:, :10], columns=["ever_married", "Residence_type"], drop_first=True)
data_X = pd.get_dummies(binary_data_x, columns=["gender", "work_type", "smoking_status"])
data_Y = dropped_data.iloc[:, 10].values.ravel()


# %%


# %%
scaler = StandardScaler()
pca = PCA()
clf = SVC()
pipeline = Pipeline([("scaler", scaler),
                     ("pca", pca),
                     ("svm", clf)])

param_grid = {
    "pca__n_components": list(range(5, 20)),
    "svm__kernel": ["linear", "rbf", "poly"]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1")
pred_Y = cross_val_predict(grid_search, data_X, data_Y, cv=10)

print("Accuracy: ", f1_score(data_Y, pred_Y))
print(classification_report(data_Y, pred_Y))

# %%
clf = tree.DecisionTreeClassifier()
scaler = StandardScaler()