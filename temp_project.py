# %% [markdown]
#  # Introduction
#  Stroke is one of the leading causes of death and disability in the United States.
#  Every year nearly 800,000 Americans suffer from stroke [[1](https://www.cdc.gov/stroke/facts.htm#:~:text=Every%2040%20seconds%2C%20someone%20in,minutes%2C%20someone%20dies%20of%20stroke.&text=Every%20year%2C%20more%20than%20795%2C000,United%20States%20have%20a%20stroke.)].
#  Of these cases, nearly 140,000
#  Americans die and another 75% of victims are left with some form of dysfunction
#  or disability [[2](https://www.frontiersin.org/articles/10.3389/fneur.2021.649088/full)].
# 
#  However, 80% of all strokes are preventable with treatment and
#  life style changes if detected early [[1](https://www.cdc.gov/stroke/facts.htm#:~:text=Every%2040%20seconds%2C%20someone%20in,minutes%2C%20someone%20dies%20of%20stroke.&text=Every%20year%2C%20more%20than%20795%2C000,United%20States%20have%20a%20stroke.)].
#  Stroke detection has become increasingly
#  important in trying to find factors that predict a stroke before it even happens.
#  By performing data exploration, feature engineering, and machine learning modeling
#  on a dataset containing many records of 10 key features the could predict stroke
#  events, we hope to produce a machine learning model for predicting stroke events
#  early to help curtail preventable deaths and ailments caused by stroke.

# %%
# import warnings
# warnings.simplefilter("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from imblearn.pipeline import Pipeline   # required to use SMOTE
from imblearn.over_sampling import SMOTE # See Piazza Post @401

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

# %% [markdown]
#  # Data Cleaning, Preparation, and Initial Feature Engineering

# %% [markdown]
#  The data was aquired from a confidential source on [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).

# %%
raw_data = pd.read_csv("healthcare-dataset-stroke-data.csv")
raw_data = raw_data.drop("id", axis=1)
raw_data.shape

# %% [markdown]
#  Here we can see the first 10 records in our raw dataset prior to any data
#  cleaning or feature engineering. Note that we threw out the ID column when we read in the data.

# %%
raw_data.head(10)


# %% [markdown]
#  ## Unknown and Missing Values

# %% [markdown]
#  Before we are able to model stroke, we have to clean our data first.
#  It appears that we have two columns with NaNs and Unknowns: `bmi` and `smoking_status`.
#  We must handle these values by either removing them or imputing them.
#  Let's first explore the `smoking_status` column.

# %% [markdown]
#  ### Smoking Status `Unknowns`

# %% [markdown]
#  The missing value is labeled as "Unknown". We will need to look at the distribution of all
#  of the labels before we are able to do anything.

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
#  Since over 30% of the data has an "unknown" smoking status, attempting to impute these values will be
#  a significant challenge. Smoking is a [well-known](https://www.cdc.gov/tobacco/campaign/tips/diseases/heart-disease-stroke.html)
#  stroke risk, and since this data is categorical, imputing these values may end up heavily
#  biasing the data toward smokers/non-smokers.
# 
#  We also are unable to ascertain where and how this data was collected, which would have informed us
#  on how to handle this data with more certainty.
# 
#  In a non-coursework setting, we would be spending significant amounts of time on trying different feature engineering
#  techniques to handle this data. However, for the purposes of this project, we decided to move forward
#  by simply dropping the unknown records. We still have thousands of other records to rely on.
# 
#  TODO (optional?) we can display summary statistics/distribution before and after dropping.

# %%
dropped_data = raw_data[raw_data.smoking_status != "Unknown"]
print("After removing the records with \"Unknown\" as their smoking status, we are left with", dropped_data.shape[0], "records.")

# %% [markdown]
#  #### Integer Encoding for `smoking_status`

# %% [markdown]
#  Now that we have dropped the unknown data, we are now left with 3 labels: `smokes`, `formerly smoked`, and
#  `never smoked`. These three labels have an ordinal property to them, where `never smoked` is "better than"
#  `formerly smoked`, which is "better than" `smokes`. To encode this property into our data,
#  we will replace these values with 0, 1, and 2, respectively, creating a smoking metric where the higher value
#  indicates a "worse smoking habit".
# 
#  While this encodes that ordinal property, it subtly implies that `smokes`, `formerly smoked`, and `never smoked`
#  are only "different" from each other on this metric by 1. `smokes` is only "2-worse" than `never smoked`, and
#  `formerly smoked` is exactly the midpoint between `smokes` and `never smoked`.
# 
#  This is yet another area we could explore in the feature-engineering iteration process, assigning different
#  values to each label and seeing how they affect the labels. Of course, this only affects models that are
#  distance-based, like K-Nearest Neighbors and Neural Networks. This should not affect models like Decision Trees.

# %%
dropped_data.loc[:, 'smoking_status'] = dropped_data.loc[:, 'smoking_status'].replace(["smokes"], 2)
dropped_data.loc[:, 'smoking_status'] = dropped_data.loc[:,'smoking_status'].replace(["formerly smoked"], 1)
dropped_data.loc[:, 'smoking_status'] = dropped_data.loc[:,'smoking_status'].replace(["never smoked"], 0)

# %% [markdown]
#  ### BMI NaNs
#  One natural question to ask is to figure out what proportion of the entries has a NaN in `bmi`.

# %%
num_bmi_nan = dropped_data["bmi"].isnull().sum()
total_entries = len(dropped_data.index)
num_bmi_nan / total_entries * 100

# %% [markdown]
#  Only 3.9% of the data contains a NaN in the data. Since there aren't that many, it seems
#  reasonable to impute the BMI with the other values.
# 
#  However, before we do so, this may introduce some bias toward stroke or non-stroke patients.
# 
#  How many of the missing values are classified as stroke patients compared to non-stroke patients?

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
#  It appears that there are more missing BMI values for stroke patents than non-stroke patients,
#  proportionally. Since these proportions are different, we will impute the BMI for stroke and
#  non-stroke patients separately.
# 
#  By what metric should we impute the BMI? If the BMI has a significant skew, the metric by
#  which we impute this data may impact our results.

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
#  The histograms and boxplots shows that the data seems to be partially right-skewed.
#  Using the mean for imputation may lead to less-than-ideal results. Thus, we shall use
#  the median to impute the missing BMI values.
# 
#  Since only 3.5% of the non-stroke patients have a NaN BMI, imputing the NaNs for the BMI
#  before we run nested cross-validations should not affect the overall performance of the models,
#  despite the data leakage. However, with 10.9% of the stroke-patients having NaN BMIs, this would
#  be a significant instance of data leakage.
# 
#  For the purposes of this project, we opted to impute the data before we split the training and test
#  data sets (see Piazza Post @395). Additionally, as you will see below, the median for both stroke and
#  non-stroke patients end up being rather close, so the data leakage should not affect it too badly.
#  Had we used the mean, we could not have gone away with this and would have had to resort to
#  manual cross-validation and imputation.

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
#  ## One-Hot Encoding

# %% [markdown]
#  Since we are performing various different types of modeling, where some cannot handle categorical data types,
#  we will be doing one-hot encoding on the categorical features.

# %% [markdown]
#  ### Binary Features

# %%
print(imputed_data["ever_married"].unique())
print(imputed_data["Residence_type"].unique())

# %% [markdown]
#  `ever_married`, and `Residence_type` are both binary categorical features. We can use Pandas'
#  `get_dummies()` function with `drop_first = True` to limit the dimensionality.

# %%
# The features and class columns are separately handled due to sklearn's API.
binary_data_X = pd.get_dummies(imputed_data.iloc[:, :10], columns=["ever_married", "Residence_type"], drop_first=True)
binary_data_X.head()

# %% [markdown]
#  ### Non-binary Features

# %%
# TODO consider dropping the gender_other column, since gender identity probably doesn't affect much in the way
# TODO of stroke prediction. also, there are very few gender_others.
print(binary_data_X["gender"].unique()) 
print(binary_data_X["work_type"].unique())

# %% [markdown]
#  Since `gender` and `work_type` are not binary features, we can use the same function, but with
#  `drop_first = False`.

# %%
data_X = pd.get_dummies(binary_data_X, columns=["gender", "work_type"], drop_first=False)
data_Y = imputed_data.iloc[:, 10].values.ravel()
data = pd.concat([data_X, imputed_data.iloc[:, 10]], axis=1)

# %% [markdown]
#  ## Additional Data Exploration

# %% [markdown]
#  Now that the data has been cleaned, we can look at some of its features and think of potential
#  additional feature engineering techniques.

# %% [markdown]
#  ### Correlation and Scatter Matrix for Age, Glucose Level, BMI, and Stroke

# %% [markdown]
#  These features are originally numeric, not categorical or binary.

# %%
print(data.iloc[:, [0, 3, 4, 16]].corr())
pd.plotting.scatter_matrix(data.iloc[:, [0, 3, 4, 16]], figsize=(15, 15))

# %% [markdown]
#  None of these features have a strong correlation with one another, so we cannot justify dropping one of
#  these features.
#  TODO comment on the scatter matrix. Cite how age and blood sugar changes together, and the data doesn't show that
#  TODO which then implies that the blood sugar was taken at random times...
# 
#  Also, there are no clear, direct correlation between any of these features with stroke.

# %% [markdown]
#  ### Correlation in Gender

# %%
print(data.iloc[:, [8, 9, 10, 16]].corr())

# %% [markdown]
#  As seen above, the `gender_Female` and `gender_Male` features obviously have strong negative correlation
#  with each other. However, we should not necessarily drop one or the other. Women appear to have a
#  [higher risk](https://www.ahajournals.org/doi/10.1161/CIRCRESAHA.121.319915#:~:text=In%20the%20United%20States%2C%20the,55%2Dyear%2Dold%20individual.&text=Stroke%20is%20more%20likely%20to,heart%20disease%20is%20more%20common.)
#  of having a stroke than men. Of course, the features list "gender" and not "sex", but people who do not
#  conform to gender norms often experience [higher levels of stress](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4201643/)
#  and stress is a [known risk factor for stroke](https://health.clevelandclinic.org/stressed-work-may-higher-risk-stroke/).
# 
#  Thus, treating each gender label in this data separately, at least initially, seems to be a good idea.

# %% [markdown]
#  ### Correlation of Work Types, Residences, and Stroke

# %%
print(data.iloc[:, [7, 11, 12, 13, 14, 15, 16]].corr())

# %% [markdown]
#  Based on the correlation matrix above, there are 2 feature pairs that are moderately correlated
#  with one another. `private` sector jobs appear to share decent correlations with `government` and `self-employment`.
#  One option to reduce the dimensionality of the data set would have been to group some of these features together.
#  However, this is probably not a great idea since, like the stress cited before, [work stress](https://www.aan.com/PressRoom/Home/PressRelease/1412)
#  is also a known risk factor for strokes. Different work types have different levels of stress, so
#  combining them would be bad for the data.
# 
#  We could attempt to group these features by varying levels of "work stress", but these work types are too broad
#  to reasonably group them like that.
# 
#  Like before, there are no clear, direct correlations between any of these features with stroke.

# %% [markdown]
#  # Initial Evaluation of Various Models

# %% [markdown]
#  Before we go into implementing the individual models and computing their performance on this data,
#  we must first address how we are going to evaluate each model.

# %%
stroke_data = data[data["stroke"] == 1]
nonstroke_data = data[data["stroke"] == 0]

stroke_type = ["stroke", "nonstroke"]
stroke_len = [len(stroke_data), len(nonstroke_data)]

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(stroke_type, stroke_len)
ax.set_title("Stroke vs Nonstroke Patients Counts in Data")

# %% [markdown]
#  As shown above, there is a significant class imbalance in our data set. If we decided to use the normal
#  k-folds procedure, there is a significant chance that some training splits will only have
#  non-stroke patients. By training on that imbalanced data, the models will be forced to only predict "nonstroke".
# 
#  In order to resolve this train/test splitting problem, we can use `StratifiedKFold()`, which will preserve
#  the relative percentage of stroke/nonstroke records in each fold. By default, this method uses `k=5` folds.
# 
#  Another option is to use SMOTE, to oversample the stroke records and artificially create new ones. Due to
#  the low number of stroke records we have available in the data, we will use Imbalanced-Learn's `SMOTE()` sampling
#  method instead of the `StratifiedKFold()`
# 
# 
#  Additionally, we must carefully consider the scoring metric by which we will grade each model's performance,
#  during and after nested cross-validation. Using accuracy is not useful, since each model can simply predict
#  "nonstroke" and get extremely high accuracy scores. The obvious solution would be to use the f1-scoring metric
# 
#  Having false-positives has a very low "societal" cost. If the model falsely predicts that a patient will have
#  a stroke (or at the very least a high risk of one), that would be okay, since the patient will merely have
#  to try and change some of their lifestyles, although this may inconvenience them.
# 
#  However, false-negatives has a massive cost. If the model falsely predicts that a patient will not have a
#  stroke, that patient may die from that stroke, or be left with lifelong symptoms.
# 
#  As such, in order to minimize false-negatives, regardless of the false-positives, we could use `recall`.
#  However, this may incentivize some algorithms to simply evaluate everything as stroke, which is the opposite of
#  what we want.
# 
#  In order to balance both of these options, we will stick with the f1-score.

# %% [markdown]
performance_dict = {}

# %% [markdown]
#  ## Decision Trees

# %% [markdown]
#  Decision Trees can handle non-scaled data and is not affected by the curse of dimensionality. Thus,
#  we can simply create a normal `DecisionTreeClassifier()` object, pass in a variety of hyperparameters,
#  and evaluate its performance.
#
# This model took 37.3 seconds to run on the CS lab machines.

# %%
smote = SMOTE() # TODO check if setting random state or not affects the outcome
clf = DecisionTreeClassifier()
pipeline = Pipeline([("sampler", smote),
                     ("tree", clf)])

params = {
    "tree__max_depth": list(range(10, 24, 5)),
    "tree__min_samples_leaf": [5, 10, 15, 20],
    "tree__max_features": list(range(5, 18, 2))
}

grid_search = GridSearchCV(pipeline, params, cv=5, scoring="f1")
pred_Y = cross_val_predict(grid_search, data_X, data_Y, cv=5)
general_score = f1_score(data_Y, pred_Y)
print("Generalization F1 Score: ", general_score)
print("Confusion Matrix: \n", confusion_matrix(data_Y, pred_Y))
print("Classification Report: \n", classification_report(data_Y, pred_Y))

performance_dict['DecisionTreeClassifier'] = general_score

# %% [markdown]
# As seen above, the decision tree classifier performed rather poorly. Although
# it achieved a high accuracy, it achieves that accuracy by guessing most of the
# records to be non-stroke (as we will see rest of the models).

# %% [markdown]
#  ## Naive Bayes

# %% [markdown]
#  Much like Decision Trees, Naive Bayes avoids the problems with the curse of dimensionality. Additionally, since
#  we have already converted our features into numerical values, we can simply use sklearn's `GaussianNB()` classifier.
#  Also, since there are no notable hyperparmeters to consider (at least from our course), we can simply
#  use `cross_val_predict()` to evaluate this model.
#
# This model took 0.1 seconds to run on the CS lab machines.

# %%
smote = SMOTE()
clf = GaussianNB()
pipeline = Pipeline([("sampler", smote),
                     ("clf", clf)])

pred_Y = cross_val_predict(pipeline, data_X, data_Y, cv=5)

general_score = f1_score(data_Y, pred_Y)
print("Generalization F1 Score: ", general_score)
print("Confusion Matrix: \n", confusion_matrix(data_Y, pred_Y))
print("Classification Report: \n", classification_report(data_Y, pred_Y))

performance_dict['GaussianNB'] = general_score

# %% [markdown]
# The result for Naive Bayes is the only noteworthy model. Although it does not
# have a remarkable f1-score, its recall on the stroke records is incredibly high
# at 88%. Of course, this resulted in a significant amount of false positives
# and a precision of only 6%. However, the number of false negatives is heavily diminished.
#
# Since this model is statistics-based, one could argue that the people with
# false positives are statistically likely to have a stroke. However, this result
# may have been from potentially redundant features.

# %% [markdown]
#  ## K-Nearest Neighbors

# %% [markdown]
#  K-Nearest Neighbors
#  uses distance calculations to classify new objects. As such, it suffers under the curse of dimensionality,
#  and we must use a method to reduce the number of dimensions, like Principle Component Analysis (PCA).
# 
#  Additionally, we must also scale the data so that features like `age` do not single-handedly determine the
#  distance calculations. This scaling is also essential for PCA.
# 
#  One option for scaling is to standardize the data based on the mean and standard deviation - converting each
#  value into a z-score (`StandardScaler()`)
#  Note that it takes a substantial amount of time to run the following, due to the shear number of distance calculations
#  required.
#
# This model took 1 minute 46 seconds to run on the CS lab machines.

# %%
smote = SMOTE()
scaler = StandardScaler()
pca = PCA()
clf = KNeighborsClassifier(n_jobs=-1) # n_jobs is set to use all processors for faster computation

pipeline = Pipeline([("sampler", smote),
                     ("scaler", scaler),
                     ("pca", pca),
                     ("knn", clf)])

params = {
    "pca__n_components": list(range(6, 17, 1)),
    "knn__n_neighbors": list(range(10, 31, 5))
}

grid_search = GridSearchCV(pipeline, params, cv=5, scoring="f1")
pred_Y = cross_val_predict(grid_search, data_X, data_Y, cv=5)
general_score = f1_score(data_Y, pred_Y)
print("Generalization F1 Score: ", general_score)
print("Confusion Matrix: \n", confusion_matrix(data_Y, pred_Y))
print("Classification Report: \n", classification_report(data_Y, pred_Y))
performance_dict['KNeighborsClassifier'] = general_score

# %% [markdown]
#  TODO make notes on KNN
#  !Additionally, PCA works by maximizing the variance in the data
#  !in each iteration. Using normalization

# %% [markdown]
#  ## Support Vector Machines

# %% [markdown]
#  Much like KNN, we need to scale and reduce the dimensionality of the data with SVMs. However, as we saw before,
#  it seems like "" is better for the data, in general, so we shall stick to "" for the rest of the models.
#
# This model took 6 minutes 43 seconds to run on the CS lab machines.
# %%
smote = SMOTE()
scaler = StandardScaler() # TODO check this
pca = PCA()
clf = SVC()
pipeline = Pipeline([("sampler", smote),
                     ("scaler", scaler),
                     ("pca", pca),
                     ("svm", clf)])

params = {
    "pca__n_components": list(range(5, 17, 1)),
    "svm__kernel": ["poly", "rbf", "sigmoid"]
}

grid_search = GridSearchCV(pipeline, params, cv=5, scoring="f1")
pred_Y = cross_val_predict(grid_search, data_X, data_Y, cv=5)
general_score = f1_score(data_Y, pred_Y)
print("Generalization F1 Score: ", general_score)
print("Confusion Matrix: \n", confusion_matrix(data_Y, pred_Y))
print("Classification Report: \n", classification_report(data_Y, pred_Y))
performance_dict['SVM'] = general_score

# %% [markdown]
#  ## Neural Networks (Multi-Layer Perceptrons)
#
# %% [markdown]
# TODO some intro to neural nets?
# This model took 9 minutes 10 seconds to run on the CS lab machines.
# %%
smote = SMOTE()
scaler = StandardScaler()
clf = MLPClassifier()
pipeline = Pipeline([("smote", smote),
                     ("scaler", scaler),
                     ("nn", clf)])

params = {
    "nn__hidden_layer_sizes": [(40,), (60,), (80,), (100,)],
    "nn__activation": ["logistic", "tanh", "relu"]
}

grid_search = GridSearchCV(pipeline, params, cv=5, scoring="f1")
pred_Y = cross_val_predict(grid_search, data_X, data_Y, cv=5)
general_score = f1_score(data_Y, pred_Y)
print("Generalization F1 Score: ", general_score)
print("Confusion Matrix: \n", confusion_matrix(data_Y, pred_Y))
print("Classification Report: \n", classification_report(data_Y, pred_Y))
performance_dict['MLPClassifier'] = general_score

# %% [markdown]
#  ## Random Forest Ensemble

# %% [markdown]
#  We can also use ensemble methods like `RandomForestClassifier()`. Much like normal decision
#  trees, we do not need to scale the data, nor do we need PCA. However, we still need a Pipeline
#  object in order to use SMOTE.
#
# This model took 8 minutes 10 seconds to run on the CS lab machines.
# %%
smote = SMOTE()
clf = RandomForestClassifier()
pipeline = Pipeline([("sampler", smote),
                     ("rfc", clf)])

params = {
    "rfc__max_depth": list(range(35, 46, 1)),
    "rfc__min_samples_leaf": list(range(8, 13, 2)),
    "rfc__max_features": ["sqrt", "log2"]
}

grid_search = GridSearchCV(pipeline, params, cv=5, scoring="f1")
pred_Y = cross_val_predict(grid_search, data_X, data_Y, cv=5)
general_score = f1_score(data_Y, pred_Y)
print("Generalization F1 Score: ", general_score)
print("Confusion Matrix: \n", confusion_matrix(data_Y, pred_Y))
print("Classification Report: \n", classification_report(data_Y, pred_Y))
performance_dict['RandomForestClassifier'] = general_score

# %% [markdown]
# ## Adaptive Boosting Ensemble

# %% [markdown]
# Another ensemble model we can use is `AdaBoostClassifier`. Since the base classifier is also decision tree stumps, we can do a similar operation to the previous ensemble method.
#
# This model took 43 seconds to run on the CS lab machines.

# %%
smote = SMOTE()
clf = AdaBoostClassifier()
pipeline = Pipeline([("sampler", smote),
                     ("adc", clf)])

params = {
    "adc__n_estimators": list(range(50, 201, 50))
}

grid_search = GridSearchCV(pipeline, params, cv=5, scoring="f1")
pred_Y = cross_val_predict(grid_search, data_X, data_Y, cv=5)
general_score = f1_score(data_Y, pred_Y)
print("Generalization F1 Score: ", general_score)
print("Confusion Matrix: \n", confusion_matrix(data_Y, pred_Y))
print("Classification Report: \n", classification_report(data_Y, pred_Y))
performance_dict['AdaBoostClassifier'] = general_score

# %%
# ## Model Performance Comparison and Summary

for key in performance_dict.keys():
    print(key, ": ", performance_dict[key])

best_model = max(performance_dict, key=performance_dict.get)
performance_dict = {}
# %% [markdown]
# As seen above, the best performing model in terms of f1-score is the random
# forest classifier. However, it only achieved an f1-score of 19%. 
#
# On the other hand, when we are considering the recall values of each model,
# the Naive Bayes model performed the best in recall, even though none of the
# models were trained using that metric.
#
# If we stopped here, we would use the Naive Bayes model as our final model.

# %% [markdown]
# # Feature Engineering Iterations and Modeling

# %% [markdown]
# Now that we have evaluated some models without feature engineering, we should try and improve upon them through feature engineering.
# We can feature engineer the glucose levels with other numeric features to incorporate some known-correlations
# in the data.

# %% [markdown]
# ## Average Glucose Level to Age Ratio

# %% [markdown]
# Studies indicate that the average glucose level in people can vary depending on age. 
# This suggests that it can be useful to calculate an "average glucose to age" ratio based on binned values. 
# Studies indicate that average glucose values tend to differ in different age groups such as from `0-6`, `6-12`, `13-19`, `20-65`, and `65+`.
# By calculating the average glucose level of each of these groups then calculating the ratio 
# relative to this average we can generate a ratio that better represents the "unusualness" of the 
# average glucose in a particular record.

# %%
data["age"].describe()
# %%
data["age"].hist()
# %% [markdown]
# Based on the fact that the minimum age for this data is 10, we decided to bin the ages into
# `0-12`, `13-19`, `20-65`, and `65+`.
# %%
group_one = 0 # TODO cite studies
count_one = 0
group_two = 0
count_two = 0
group_three = 0
count_three = 0
group_four = 0
count_four = 0

for age in data["age"]:
   if age <= 12:
      group_one += age
      count_one += 1
   elif age <= 19:
      group_two += age
      count_two += 1
   elif age <= 65:
      group_three += age
      count_three += 1
   else:
      group_four += age
      count_four += 1

group_one_avg = group_one / count_one
group_two_avg = group_two / count_two
group_three_avg = group_three / count_three
group_four_avg = group_four / count_four

glucose_age_ratio = []

for age in data["age"]:
    if age <= 12:
        glucose_age_ratio.append(age/group_one_avg)
    elif age <= 19:
        glucose_age_ratio.append(age/group_two_avg)
    elif age <= 65:
        glucose_age_ratio.append(age/group_three_avg)
    else:
        glucose_age_ratio.append(age/group_four_avg)

data["glucose_age_ratio"] = glucose_age_ratio
data_X = data.drop(columns=["stroke"])
data_Y = data["stroke"]

data_X.head()
# %%
data_X["glucose_age_ratio"].describe()
# %%
data_X["glucose_age_ratio"].hist()
# %% [markdown]
# As seen in the distribution above, our new feature seems to create a new unimodal distribution,
# where 1.0 is what is expected while values beyond a standard deviation or so can be viewed as abnormal for that age.

# %% [markdown]
# ## Diabetes Evidence

# %% [markdown]
# Studies indicate that having diabetes increases your risk of a stroke by 
# 1.5 times. Although diabetes was not a recorded feature in the dataset, using 
# the average glucose level we can feature engineer to create a binary "diabetes_evidence" 
# feature that reports if there is evidence of diabetes based on glucose level. Blood sugar tests 
# determine that a person has diabetes if they have an average gluvose level above 154. We can use this 
# threshold and determine diabetes evidence for each record based on this test.

# %%
diabetes_evidence = []
for avg_glucose_level in data["avg_glucose_level"]:
   if avg_glucose_level >= 154:
      diabetes_evidence.append(1)
   else:
      diabetes_evidence.append(0)

data["diabetes_evidence"] = diabetes_evidence
data.head()


# %% [markdown]
# ### Average Glucose Level to BMI Ratio

# %% [markdown]
# The average glucose level can also vary according to 
# BMI. Inorder to better measure a person's glucose level 
# in respect to their BMI, we can make a ratio that divides 
# a person's avg glucose level by their BMI. 

# %%
glucose_bmi_ratio = []

for index, row in data.iterrows():
   glucose_bmi_ratio.append(row["avg_glucose_level"]/row["bmi"])

data["glucose_bmi_ratio"] = glucose_bmi_ratio
data.head()

# %% [markdown]
# # Testing the Models with New Features
# %% [markdown]
# With this new feature, the following runs the same `7` models as we did before for comparison.

# %%
# Decision Tree
smote = SMOTE()
clf = DecisionTreeClassifier()
pipeline = Pipeline([("sampler", smote),
                     ("tree", clf)])

# params = {
#     "tree__max_depth": list(range(10, 24, 5)),
#     "tree__min_samples_leaf": [5, 10, 15, 20],
#     "tree__max_features": list(range(5, 18, 2))
# }

grid_search = GridSearchCV(pipeline, params, cv=5, scoring="f1")
pred_Y = cross_val_predict(grid_search, data_X, data_Y, cv=5)
general_score = f1_score(data_Y, pred_Y)
print("Generalization F1 Score: ", general_score)
print("Confusion Matrix: \n", confusion_matrix(data_Y, pred_Y))
print("Classification Report: \n", classification_report(data_Y, pred_Y))

performance_dict['DecisionTreeClassifier'] = general_score

# %%
# Naive Bayes
smote = SMOTE()
clf = GaussianNB()
pipeline = Pipeline([("sampler", smote),
                     ("clf", clf)])

pred_Y = cross_val_predict(pipeline, data_X, data_Y, cv=5)

general_score = f1_score(data_Y, pred_Y)
print("Generalization F1 Score: ", general_score)
print("Confusion Matrix: \n", confusion_matrix(data_Y, pred_Y))
print("Classification Report: \n", classification_report(data_Y, pred_Y))

performance_dict['GaussianNB'] = general_score

# %%
# K-Nearest Neighbors
smote = SMOTE()
scaler = StandardScaler()
pca = PCA()
clf = KNeighborsClassifier(n_jobs=-1) # n_jobs is set to use all processors for faster computation

pipeline = Pipeline([("sampler", smote),
                     ("scaler", scaler),
                     ("pca", pca),
                     ("knn", clf)])

# params = {
#     "pca__n_components": list(range(6, 17, 1)),
#     "knn__n_neighbors": list(range(10, 31, 5))
# }

grid_search = GridSearchCV(pipeline, params, cv=5, scoring="f1")
pred_Y = cross_val_predict(grid_search, data_X, data_Y, cv=5)
general_score = f1_score(data_Y, pred_Y)
print("Generalization F1 Score: ", general_score)
print("Confusion Matrix: \n", confusion_matrix(data_Y, pred_Y))
print("Classification Report: \n", classification_report(data_Y, pred_Y))
performance_dict['KNeighborsClassifier'] = general_score


# %%
# SVM
smote = SMOTE()
scaler = StandardScaler() # TODO check this
pca = PCA()
clf = SVC()
pipeline = Pipeline([("sampler", smote),
                     ("scaler", scaler),
                     ("pca", pca),
                     ("svm", clf)])

# params = {
#     "pca__n_components": list(range(5, 17, 1)),
#     "svm__kernel": ["poly", "rbf", "sigmoid"]
# }

grid_search = GridSearchCV(pipeline, params, cv=5, scoring="f1")
pred_Y = cross_val_predict(grid_search, data_X, data_Y, cv=5)
general_score = f1_score(data_Y, pred_Y)
print("Generalization F1 Score: ", general_score)
print("Confusion Matrix: \n", confusion_matrix(data_Y, pred_Y))
print("Classification Report: \n", classification_report(data_Y, pred_Y))
performance_dict['SVM'] = general_score

# %%
# Neural Networks
smote = SMOTE()
scaler = StandardScaler()
clf = MLPClassifier()
pipeline = Pipeline([("smote", smote),
                     ("scaler", scaler),
                     ("nn", clf)])

params = {
    "nn__hidden_layer_sizes": [(40,), (60,), (80,), (100,)],
    "nn__activation": ["logistic", "tanh", "relu"]
}

grid_search = GridSearchCV(pipeline, params, cv=5, scoring="f1")
pred_Y = cross_val_predict(grid_search, data_X, data_Y, cv=5)
general_score = f1_score(data_Y, pred_Y)
print("Generalization F1 Score: ", general_score)
print("Confusion Matrix: \n", confusion_matrix(data_Y, pred_Y))
print("Classification Report: \n", classification_report(data_Y, pred_Y))
performance_dict['MLPClassifier'] = general_score

# %%
# Random Forest
smote = SMOTE()
clf = RandomForestClassifier()
pipeline = Pipeline([("sampler", smote),
                     ("rfc", clf)])

params = {
    "rfc__max_depth": list(range(35, 46, 1)),
    "rfc__min_samples_leaf": list(range(5, 16, 1)),
    "rfc__max_features": ["sqrt", "log2"]
}

grid_search = GridSearchCV(pipeline, params, cv=5, scoring="f1")
pred_Y = cross_val_predict(grid_search, data_X, data_Y, cv=5)
general_score = f1_score(data_Y, pred_Y)
print("Generalization F1 Score: ", general_score)
print("Confusion Matrix: \n", confusion_matrix(data_Y, pred_Y))
print("Classification Report: \n", classification_report(data_Y, pred_Y))
performance_dict['RandomForestClassifier'] = general_score

# %%
# ## Model Performance Comparison

print(performance_dict)
best_model = max(performance_dict, key=performance_dict.get)
print(f"best performing model is {best_model}, with the best f1 score of {performance_dict[best_model]}")
performance_dict = {}

# %%
# scaler = StandardScaler()
# pca = PCA()
# clf = SVC()
# pipeline = Pipeline([("scaler", scaler),
#                      ("pca", pca),
#                      ("svm", clf)])

# param_grid = {
#     "pca__n_components": list(range(5, 20)),
#     "svm__kernel": ["linear", "rbf", "poly"]
# }

# grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1")
# pred_Y = cross_val_predict(grid_search, data_X, data_Y, cv=10)

# print("Accuracy: ", f1_score(data_Y, pred_Y))
# print(classification_report(data_Y, pred_Y))
# '''
# Accuracy:  0.018099547511312215
#               precision    recall  f1-score   support

#            0       0.96      1.00      0.98      4700
#            1       0.17      0.01      0.02       209

#     accuracy                           0.96      4909
#    macro avg       0.56      0.50      0.50      4909
# weighted avg       0.92      0.96      0.94      4909

# '''

# # %%
# clf = tree.DecisionTreeClassifier()
# pca = decomposition.PCA()
# std_slc = StandardScaler()
# pipe = Pipeline(steps=[('std_slc', std_slc),
#                         ('pca', pca),
#                         ('dec_tree', clf)])
# n_components = list(range(1,data_X.shape[1]+1,1))
# param_grid = dict(pca__n_components=n_components,
#                     dec_tree__criterion=['gini', 'entropy'],
#                 dec_tree__max_depth = [2,4,6,8,10,12])
# grid_search = GridSearchCV(clf, param_grid, scoring="f1")
# pred_Y = cross_val_predict(clf, data_X, data_Y)
# print("Accuracy: ", f1_score(data_Y, pred_Y))
# print(classification_report(data_Y, pred_Y))
# '''
# Accuracy:  0.12340425531914893
#               precision    recall  f1-score   support

#            0       0.96      0.95      0.96      4700
#            1       0.11      0.14      0.12       209

#     accuracy                           0.92      4909
#    macro avg       0.54      0.54      0.54      4909
# weighted avg       0.93      0.92      0.92      4909

# '''


# # %%
# clf = RandomForestClassifier(n_estimators=100)
# param_grid = dict()
# grid_search = GridSearchCV(clf, param_grid, scoring="f1")
# pred_Y = cross_val_predict(clf, data_X, data_Y)
# print("Accuracy: ", f1_score(data_Y, pred_Y))
# print(classification_report(data_Y, pred_Y))
# '''
# Accuracy:  0.018348623853211007
#               precision    recall  f1-score   support

#            0       0.96      1.00      0.98      4700
#            1       0.22      0.01      0.02       209

#     accuracy                           0.96      4909
#    macro avg       0.59      0.50      0.50      4909
# weighted avg       0.93      0.96      0.94      4909
# '''


# %%



