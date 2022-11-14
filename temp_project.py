# %% [markdown]
# # Introduction TODO
# Stroke is one of the leading causes of death and disability in the United States. 
# Every year nearly 800,000 Americans suffer from stroke [[1](https://www.cdc.gov/stroke/facts.htm#:~:text=Every%2040%20seconds%2C%20someone%20in,minutes%2C%20someone%20dies%20of%20stroke.&text=Every%20year%2C%20more%20than%20795%2C000,United%20States%20have%20a%20stroke.)]. 
# Of these cases, nearly 140,000 
# Americans die and another 75% of victims are left with some form of dysfunction 
# or disability [[2](https://www.frontiersin.org/articles/10.3389/fneur.2021.649088/full)]. 
# 
# However, 80% of all strokes are preventable with treatment and 
# life style changes if detected early [[1](https://www.cdc.gov/stroke/facts.htm#:~:text=Every%2040%20seconds%2C%20someone%20in,minutes%2C%20someone%20dies%20of%20stroke.&text=Every%20year%2C%20more%20than%20795%2C000,United%20States%20have%20a%20stroke.)].
# Stroke detection has become increasingly 
# important in trying to find factors that predict a stroke before it even happens. 
# By performing data exploration, feature engineering, and machine learning modeling
# on a dataset containing many records of 11 key features the could predict stroke
# events, we hope to produce a machine learning model for predicting stroke events
# early to help curtail preventable deaths and ailments caused by stroke.


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition


# %%
# read in data
raw_data = pd.read_csv("healthcare-dataset-stroke-data.csv")
raw_data = raw_data.drop("id", axis=1) # TODO comment on removing the ids
raw_data.shape

# %% [markdown]
# ## Data Cleaning and Preparation

# %% [markdown]
# Here we can see the first 10 records in our raw dataset prior to any data 
# cleaning or feature engineering. 

# %% 
raw_data.head(10)

# %% [markdown]
# ### Missing Values

# %% [markdown]
# Before we are able to model stroke, we have to clean our data first.
# It appears that we have two columns with NaNs and Unknowns: `bmi` and `smoking_status`.
# We must handle these values by either removing them or imputing them.
# Let's first explore the `smoking_status` column.

# %%
# printing shape of smoking status distribution as a pie graph
def get_count(df, col, val) :
  return df[df[col] == val].shape[0]

smk_lables = ['smokes', 'Unknown', 'formerly smoked', 'never smoked']
smk_cnt = []
for label in smk_lables :
  smk_cnt.append(get_count(raw_data, "smoking_status", label))

print("There are", smk_cnt[1], "records who have an \"Unknown\" smoking status.\n")

plt.title("Smoking Status Distribution")
plt.pie(smk_cnt, labels = smk_lables, autopct='%1.1f%%')
plt.show()

# %%
# dropping records with "Unknown" in smoking status field
dropped_data = raw_data[raw_data.smoking_status != "Unknown"]
print("After removing the records with \"Unkown\" as their smoking status, we are left with", dropped_data.shape[0], "records.")
# TODO justify why we are dropping the Unknowns (30% of the data)
# %% [markdown]
# #### BMI NaNs TODO move after smoking status
# One natural question to ask is to figure out what proportion of the entries has a NaN in `bmi`.

# %% 
# changing categorical representation of smoking status into numerical type
# smokes -> 2, formerly smoked -> 1, never smoked ->0
dropped_data['smoking_status'] = dropped_data['smoking_status'].replace(["smokes"], "2")
dropped_data['smoking_status'] = dropped_data['smoking_status'].replace(["never smoked"], "0")
dropped_data['smoking_status'] = dropped_data['smoking_status'].replace(["formerly smoked"], "1")

# %%
# TODO change to the data after unknowns are removed
num_bmi_nan = dropped_data["bmi"].isnull().sum()
total_entries = len(dropped_data.index)
num_bmi_nan / total_entries * 100

# %% [markdown]
# Only 3.9% of the data contains a NaN in the data. Since there aren't that many, it seems
# reasonable to impute the BMI with the other values.
#
# However, before we do so, this may introduce some bias toward stroke or non-stroke patients.
# 
# How many of the missing values are classified as stroke patients compared to non-stroke patients?

# raw_data.drop(raw_data[raw_data['smoking_status'] == "Unknown"].index, inplace = True)

# %%
raw_stroke_bmi_data = dropped_data[dropped_data["stroke"] == 1]["bmi"]
raw_nonstroke_bmi_data = dropped_data[dropped_data["stroke"] == 0]["bmi"]

num_stroke_bmi_nan = raw_stroke_bmi_data.isnull().sum()
num_nonstroke_bmi_nan = raw_nonstroke_bmi_data.isnull().sum()

total_stroke_entries = len(raw_stroke_bmi_data.index)
total_nonstroke_entries = len(raw_nonstroke_bmi_data.index)

print("# of NaN BMI values in stroke:", num_stroke_bmi_nan)
print("# of Stroke Patients:", total_stroke_entries)
print("Percentage of NaN BMI in Stroke Patients:", num_stroke_bmi_nan / total_stroke_entries * 100, "\n")
print("# of NaN BMI values in nonstroke:", num_nonstroke_bmi_nan)
print("# of Nonstroke Patients:", total_nonstroke_entries)
print("Percentage of NaN BMI in Nonstroke Patients:", num_nonstroke_bmi_nan / total_nonstroke_entries * 100, "\n")

# %% [markdown]
# It appears that there are more missing BMI values for stroke patents than non-stroke patients, 
# proportionally. Since these proportions are different, we will impute the BMI for stroke and 
# non-stroke patients separately.
#
# By what metric should we impute the BMI? If the BMI has a significant skew, the metric by
# which we impute this data may impact our results.

# %%
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle("BMI Histograms without NaNs")

ax1.set_title("Stroke Patients") # TODO horizontal axes label
ax1.hist(raw_stroke_bmi_data[~np.isnan(raw_stroke_bmi_data)])

ax2.set_title("Non-Stroke Patients")
ax2.hist(raw_nonstroke_bmi_data[~np.isnan(raw_nonstroke_bmi_data)])

# %%
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle("BMI Box Plots without NaNs")

ax1.set_title("Stroke Patients")
ax1.boxplot(raw_stroke_bmi_data[~np.isnan(raw_stroke_bmi_data)])

ax2.set_title("Non-Stroke Patients")
ax2.boxplot(raw_nonstroke_bmi_data[~np.isnan(raw_nonstroke_bmi_data)])

# %% [markdown]
# The histograms and boxplots shows that the data seems to be partially right-skewed.
# Using the mean for imputation may lead to less-than-ideal results. Thus, we shall use
# the median to impute the missing BMI values.

# %%
stroke_median = raw_stroke_bmi_data.median()
nonstroke_median = raw_nonstroke_bmi_data.median()
print("BMI Median for Stroke Patients: ", stroke_median)
print("BMI Median for Nonstroke Patients: ", nonstroke_median)

imputed_stroke_data = dropped_data[dropped_data["stroke"] == 1].fillna(stroke_median)
imputed_nonstroke_data = dropped_data[dropped_data["stroke"] == 0].fillna(nonstroke_median)
data = pd.concat([imputed_stroke_data, imputed_nonstroke_data])
data.head(10)

# %%
# TODO print out unique values of smoking status
# TODO print out unique values of the other categories (binary and multi-type) when we are preparing it

# TODO after cleaning, compute correlation to find potentially irrelevant features

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
'''
Accuracy:  0.018099547511312215
              precision    recall  f1-score   support

           0       0.96      1.00      0.98      4700
           1       0.17      0.01      0.02       209

    accuracy                           0.96      4909
   macro avg       0.56      0.50      0.50      4909
weighted avg       0.92      0.96      0.94      4909

'''

# %%
clf = tree.DecisionTreeClassifier()
pca = decomposition.PCA()
std_slc = StandardScaler()
pipe = Pipeline(steps=[('std_slc', std_slc),
                        ('pca', pca),
                        ('dec_tree', clf)])
n_components = list(range(1,data_X.shape[1]+1,1))
param_grid = dict(pca__n_components=n_components,
                    dec_tree__criterion=['gini', 'entropy'],
                dec_tree__max_depth = [2,4,6,8,10,12])
grid_search = GridSearchCV(clf, param_grid, scoring="f1")
pred_Y = cross_val_predict(clf, data_X, data_Y)
print("Accuracy: ", f1_score(data_Y, pred_Y))
print(classification_report(data_Y, pred_Y))
'''
Accuracy:  0.12340425531914893
              precision    recall  f1-score   support

           0       0.96      0.95      0.96      4700
           1       0.11      0.14      0.12       209

    accuracy                           0.92      4909
   macro avg       0.54      0.54      0.54      4909
weighted avg       0.93      0.92      0.92      4909

'''


# %%
clf = RandomForestClassifier(n_estimators=100)
param_grid = dict()
grid_search = GridSearchCV(clf, param_grid, scoring="f1")
pred_Y = cross_val_predict(clf, data_X, data_Y)
print("Accuracy: ", f1_score(data_Y, pred_Y))
print(classification_report(data_Y, pred_Y))
'''
Accuracy:  0.018348623853211007
              precision    recall  f1-score   support

           0       0.96      1.00      0.98      4700
           1       0.22      0.01      0.02       209

    accuracy                           0.96      4909
   macro avg       0.59      0.50      0.50      4909
weighted avg       0.93      0.96      0.94      4909
'''
