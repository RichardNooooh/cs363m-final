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
# on a dataset containing many records of 10 key features the could predict stroke
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

# %% [markdown]
# # Data Cleaning, Preparation, and Initial Feature Engineering

# %%
raw_data = pd.read_csv("healthcare-dataset-stroke-data.csv")
raw_data = raw_data.drop("id", axis=1)
raw_data.shape

# %% [markdown]
# Here we can see the first 10 records in our raw dataset prior to any data 
# cleaning or feature engineering. Note that we threw out the ID column when we read in the data.

# %% 
raw_data.head(10)

# %% [markdown]
# ## Missing Values

# %% [markdown]
# Before we are able to model stroke, we have to clean our data first.
# It appears that we have two columns with NaNs and Unknowns: `bmi` and `smoking_status`.
# We must handle these values by either removing them or imputing them.
# Let's first explore the `smoking_status` column.

# %% [markdown]
# ### Smoking Status `Unknowns`

# %% [markdown]
# The missing value is labeled as "Unknown". We will need to look at the distribution of all
# of the labels before we are able to do anything.

# %%
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

# %% [markdown]
# Since over 30% of the data has an "unknown" smoking status, attempting to impute these values will be
# a significant challenge. Smoking is a [well-known](https://www.cdc.gov/tobacco/campaign/tips/diseases/heart-disease-stroke.html)
# stroke risk, and since this data is categorical, imputing these values may end up heavily
# biasing the data toward smokers/non-smokers.
# 
# We also are unable to ascertain where and how this data was collected, which would have informed us
# on how to handle this data with more certainty.
#
# In a non-coursework setting, we would be spending significant amounts of time on trying different feature engineering
# techniques to handle this data. However, for the purposes of this project, we decided to move forward
# by simply dropping the unknown records. We still have thousands of other records to rely on.
# 

# TODO (optional?) we can display summary statistics/distribution before and after dropping.

# %%
dropped_data = raw_data[raw_data.smoking_status != "Unknown"]
print("After removing the records with \"Unknown\" as their smoking status, we are left with", dropped_data.shape[0], "records.")

# %% [markdown]
# #### Integer Encoding for `smoking_status`

# %% [markdown]
# Now that we have dropped the unknown data, we are now left with 3 labels: `smokes`, `formerly smoked`, and 
# `never smoked`. These three labels have an ordinal property to them, where `never smoked` is "better than"
# `formerly smoked`, which is "better than" `smokes`. To encode this property into our data,
# we will replace these values with 0, 1, and 2, respectively, creating a smoking metric where the higher value
# indicates a "worse smoking habit".
# 
# While this encodes that ordinal property, it subtly implies that `smokes`, `formerly smoked`, and `never smoked`
# are only "different" from each other on this metric by 1. `smokes` is only "2-worse" than `never smoked`, and
# `formerly smoked` is exactly the midpoint between `smokes` and `never smoked`.
#
# This is yet another area we could explore in the feature-engineering iteration process, assigning different
# values to each label and seeing how they affect the labels. Of course, this only affects models that are
# distance-based, like K-Nearest Neighbors and Neural Networks. This should not affect models like Decision Trees.

# %% 
dropped_data.loc[:, 'smoking_status'] = dropped_data.loc[:, 'smoking_status'].replace(["smokes"], 2)
dropped_data.loc[:, 'smoking_status'] = dropped_data.loc[:,'smoking_status'].replace(["formerly smoked"], 1)
dropped_data.loc[:, 'smoking_status'] = dropped_data.loc[:,'smoking_status'].replace(["never smoked"], 0)

# %% [markdown]
# ### BMI NaNs
# One natural question to ask is to figure out what proportion of the entries has a NaN in `bmi`.
# %%
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

# %%
# TODO what to do with this commented line?.... vvvv
# raw_data.drop(raw_data[raw_data['smoking_status'] == "Unknown"].index, inplace = True)

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
#
# Since only 3.5% of the non-stroke patients have a NaN BMI, imputing the NaNs for the BMI
# before we run nested cross-validations should not affect the overall performance of the models,
# despite the data leakage. However, with 10.9% of the stroke-patients having NaN BMIs, this would
# be a significant instance of data leakage.
# 
# For the purposes of this project, we opted to impute the data before we split the training and test
# data sets (see Piazza Post @395). Additionally, as you will see below, the median for both stroke and
# non-stroke patients end up being rather close, so the data leakage should not affect it too badly.
# Had we used the mean, we could not have gone away with this and would have had to resort to
# manual cross-validation and imputation.
# %%
stroke_median = raw_stroke_bmi_data.median()
nonstroke_median = raw_nonstroke_bmi_data.median()
print("BMI Median for Stroke Patients: ", stroke_median)
print("BMI Median for Nonstroke Patients: ", nonstroke_median)

imputed_stroke_data = dropped_data[dropped_data["stroke"] == 1].fillna(stroke_median)
imputed_nonstroke_data = dropped_data[dropped_data["stroke"] == 0].fillna(nonstroke_median)
imputed_data = pd.concat([imputed_stroke_data, imputed_nonstroke_data])
imputed_data.head(10)

# %% [markdown]
# ## One-Hot Encoding

# %% [markdown]
# Since we are performing various different types of modeling, where some cannot handle categorical data types,
# we will be doing one-hot encoding on the categorical features.

# %% [markdown]
# ### Binary Features

# %%
print(imputed_data["ever_married"].unique())
print(imputed_data["Residence_type"].unique())

# %% [markdown]
# `ever_married`, and `Residence_type` are both binary categorical features. We can use Pandas'
# `get_dummies()` function with `drop_first = True` to limit the dimensionality.

# %%
# The features and class columns are separately handled due to sklearn's API.
binary_data_X = pd.get_dummies(imputed_data.iloc[:, :10], columns=["ever_married", "Residence_type"], drop_first=True)
binary_data_X.head()

# %% [markdown]
# ### Non-binary Features

# %%
# TODO consider dropping the gender_other column, since gender identity probably doesn't affect much in the way
# TODO of stroke prediction. also, there are very few gender_others.
print(binary_data_X["gender"].unique()) 
print(binary_data_X["work_type"].unique())

# %% [markdown]
# Since `gender` and `work_type` are not binary features, we can use the same function, but with
# `drop_first = False`.

# %%
data_X = pd.get_dummies(binary_data_X, columns=["gender", "work_type"], drop_first=False)
data_Y = imputed_data.iloc[:, 10].values.ravel()
data = pd.concat([data_X, imputed_data.iloc[:, 10]], axis=1)

# %% [markdown]
# ## Additional Data Exploration

# %% [markdown]
# Now that the data has been cleaned, we can look at some of its features and think of potential
# additional feature engineering techniques.

# %% [markdown]
# ### Correlation and Scatter Matrix for Age, Glucose Level, BMI, and Stroke

# %% [markdown]
# These features are originally numeric, not categorical or binary.

# %%
print(data.iloc[:, [0, 3, 4, 16]].corr())
pd.plotting.scatter_matrix(data.iloc[:, [0, 3, 4, 16]], figsize=(15, 15))

# %% [markdown]
# None of these features have a strong correlation with one another, so we cannot justify dropping one of
# these features.
# TODO comment on the scatter matrix. Cite how age and blood sugar changes together, and the data doesn't show that
# TODO which then implies that the blood sugar was taken at random times...

# %% [markdown]
# ### Correlation in Gender and Stroke

# %%
print(data.iloc[:, [8, 9, 10, 16]].corr())

# %% [markdown]
# As seen above, the `gender_Female` and `gender_Male` features obviously have strong negative correlation
# with each other. However, we should not necessarily drop one or the other. Women appear to have a
# [higher risk](https://www.ahajournals.org/doi/10.1161/CIRCRESAHA.121.319915#:~:text=In%20the%20United%20States%2C%20the,55%2Dyear%2Dold%20individual.&text=Stroke%20is%20more%20likely%20to,heart%20disease%20is%20more%20common.)
# of having a stroke than men. Of course, the features list "gender" and not "sex", but people who do not
# conform to gender norms often experience [higher levels of stress](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4201643/)
# and stress is a [known risk factor for stroke](https://health.clevelandclinic.org/stressed-work-may-higher-risk-stroke/).
#
# Thus, treating each gender label in this data separately, at least initially, seems to be a good idea.

# %% [markdown]
# ### Correlation of Work Types, Residences, and Stroke

# %%
print(data.iloc[:, [7, 11, 12, 13, 14, 15, 16]].corr())

# %% [markdown]
# Based on the correlation matrix above, there are 2 feature pairs that are moderately correlated
# with one another. `private` sector jobs appear to share decent correlations with `government` and `self-employment`.
# One option to reduce the dimensionality of the data set would have been to group some of these features together.
# However, this is probably not a great idea since, like the stress cited before, [work stress](https://www.aan.com/PressRoom/Home/PressRelease/1412)
# is also a known risk factor for strokes. Different work types have different levels of stress, so
# combining them would be bad for the data.
#
# We could attempt to group these features by varying levels of "work stress", but these work types are too broad
# to reasonably group them like that.

# %% [markdown]
# # Modeling

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
