import pickle
from joblib import Memory
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
import Preprocessing_Class as pp
import time
import Visualizations_Classification
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv('hotel-classification-dataset.csv', parse_dates=['Review_Date'], dayfirst=True)

# Visualization Plots
Visualizations_Classification.Plots(data)

# Split data to X & Y
X = data.drop(columns=['Reviewer_Score'], axis=1)  # Features
Y = data['Reviewer_Score']  # Label

# for Pickels
models = {}


# feature encoding Y column (Reviewer_Score)
le = LabelEncoder()
Y_encod = le.fit_transform(Y)
models['LabelEncoder Y'] = le


# Split to Train & Test
X_train, X_test, y_train, y_test = train_test_split(X, Y_encod, test_size=0.20, shuffle=True, random_state=10)

pp.nulls_X(X_train)
#before X_train & X_test preprocessing
X_train , X_test = pp.fill_null(X_train ,X_test)

# Preprocessing
X_train = pp.preprocessing(X_train)
X_test = pp.preprocessing(X_test)


# call this function after X_train & X_test preprocessing

cols = X_train.select_dtypes(include=["object"]).columns.tolist()
X_train ,X_test = pp.Feature_Encoder(X_train, X_test,cols, models)

# access the LabelEncoder(X_Data)
lbl_encoder = models['label encoder']

dir = 'preprocessing_file'
mem = Memory(dir)
pre = mem.cache(pp.preprocessing)


X_train, X_test = pp.scaler(X_train, X_test,models)

X_train_fs, X_test_fs = pp.select_features(X_train, y_train, X_test,models)



print('--------------------<LogisticRegression>--------------------------')
lr = LogisticRegression(solver='newton-cg', penalty='l2', C=1)

start_train_lr = time.time()
lr.fit(X_train_fs, y_train)
models['logistic_regression'] = lr
end_train_lr = time.time()

start_test_lr = time.time()
y_pred_log = lr.predict(X_test_fs)
end_test_lr = time.time()

training_time_lr = end_train_lr - start_train_lr
test_time_lr = end_test_lr - start_test_lr

accuracy_log = accuracy_score(y_test, y_pred_log, normalize=True)
# accuracy
print("LogisticRegression test Accuracy", accuracy_log * 100)




print('--------------------<DecisionTreeClassifier>--------------------------')
dtc = DecisionTreeClassifier(min_samples_split=20, min_samples_leaf=25, max_depth=9)

start_train_dt = time.time()
dtc.fit(X_train_fs, y_train)

models['decision_tree_classifier'] = dtc

end_train_dt = time.time()
start_test_dt = time.time()
y_pred_tree = dtc.predict(X_test_fs)
end_test_dt = time.time()

training_time_dt = end_train_dt - start_train_dt
test_time_dt = end_test_dt - start_test_dt

accuracy_tree = accuracy_score(y_test, y_pred_tree, normalize=True)
# accuracy
print("DecisionTreeClassifier test Accuracy", accuracy_tree * 100)


print('--------------------<RandomForestClassifier>--------------------------')
rfc = RandomForestClassifier(n_estimators=50, max_depth=9, min_samples_leaf=25, max_features=5)

start_train_rf = time.time()
rfc.fit(X_train_fs, y_train)
models['RandomForestClassifier'] = rfc
end_train_rf = time.time()

start_test_rf = time.time()
y_pred_forest = rfc.predict(X_test_fs)
end_test_rf = time.time()

training_time_rf = end_train_rf - start_train_rf
test_time_rf = end_test_rf - start_test_rf

accuracy_forest = accuracy_score(y_test, y_pred_forest, normalize=True)
# accuracy
print("RandomForestClassifier test Accuracy", accuracy_forest * 100)




print('--------------------<AdaBoostClassifier>--------------------------')
dt_classifier = DecisionTreeClassifier(min_samples_split=20, min_samples_leaf=25, max_depth=9, criterion='entropy')
adaboost_classifier = AdaBoostClassifier(dt_classifier, n_estimators=50, learning_rate=0.01, random_state=42)

start_train_ada = time.time()
adaboost_classifier.fit(X_train_fs, y_train)

models['AdaBoostClassifier'] = adaboost_classifier

end_train_ada = time.time()

start_test_ada = time.time()
predictions_ada = adaboost_classifier.predict(X_test_fs)
end_test_ada = time.time()

training_time_ada = end_train_ada - start_train_ada
test_time_ada = end_test_ada - start_test_ada

accuracy_ada = accuracy_score(y_test, predictions_ada, normalize=True)
print("AdaBoostClassifier test Accuracy:", accuracy_ada * 100)



with open("hotel_models.pkl", "wb") as f:
    pickle.dump(models, f)


# Training Time Comparison
training_times = [training_time_lr, training_time_dt, training_time_rf, training_time_ada]
model_names = ['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier','AdaBoostClassifier']
plt.bar(model_names, training_times)
plt.title('Training Time Comparison')
plt.ylabel('Training Time (seconds)')
plt.show()


# Test Time Comparison
test_times = [test_time_lr, test_time_dt, test_time_rf, test_time_ada]
model_names = ['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'AdaBoostClassifier']
plt.bar(model_names, test_times)
plt.title('Test Time Comparison')
plt.ylabel('Test Time (seconds)')
plt.show()


# MLA_compare
MLA = [lr, dtc, rfc, adaboost_classifier]
MLA_columns = []
MLA_compare = pd.DataFrame(columns=MLA_columns)
row_index = 0

for alg in MLA:
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Test Accuracy'] = round(alg.score(X_test_fs, y_test), 4)
    row_index += 1
MLA_compare.sort_values(by=['MLA Test Accuracy'], ascending=False, inplace=True)

plt.subplots(figsize=(15, 6))
sns.barplot(x="MLA Name", y="MLA Test Accuracy", data=MLA_compare, palette='hot', edgecolor=sns.color_palette('dark'))
plt.xticks(rotation=90)
plt.title('MLA Test Accuracy Comparison')
plt.show()

# Use the saved models By pickle

with open("hotel_models.pkl", "rb") as f:
    models = pickle.load(f)


# access the logistic regression classifier model
lr_model = models['logistic_regression']
# access the decision tree classifier model
dtc_model = models['decision_tree_classifier']
# access the RandomForestClassifier classifier model
rfc_model = models['RandomForestClassifier']
# access the AdaBoostClassifier classifier model
adaboost_classifier_model = models['AdaBoostClassifier']
# access the Standard Scaler
scalerx_train = models['Standard Scaler']
# access the SelectKBest
fs = models['SelectKBest']
# access the LabelEncoder
le_Ytest = models['LabelEncoder Y']

datafile = input("Enter the test data : " )
data_test = pd.read_csv(datafile + '.csv', parse_dates=['Review_Date'], dayfirst=True)

x_data_test =data_test.drop(columns=['Reviewer_Score'], axis=1)  # Features
y_data_test = data_test['Reviewer_Score']

# in file test data
with open('column_info.plk', 'rb') as sa:
    column_info = pickle.load(sa)

for col in x_data_test.columns:
    if (x_data_test[col].isnull().sum() != 0):
        if column_info[col]['type'] == 'categorical':
            x_data_test[col] = x_data_test[col].fillna(column_info[col]['value'])
        else:
            x_data_test[col] = x_data_test[col].fillna(column_info[col]['value'])

print(x_data_test.isna().sum())

x_data_test = pre(x_data_test)

# this code in the test data after X_test_data preprocessing

cols = x_data_test.select_dtypes(include=["object"]).columns.tolist()
for col in cols:
    encoding_dict = {label: index for index, label in enumerate(lbl_encoder.classes_)}
    unknown_val = len(encoding_dict) # Assign a value to represent "unknown"
    x_data_test[col] = [encoding_dict.get(label, unknown_val) for label in x_data_test[col]]

y_data_test_encod = le_Ytest.transform(y_data_test)

X_test_scaled = scalerx_train.transform(x_data_test)

X_data_test_fs = fs.transform(X_test_scaled)

# Models

print('--------------------<LogisticRegression>--------------------------')
y_pred_log = lr_model.predict(X_data_test_fs)
accuracy_log = accuracy_score(y_data_test_encod, y_pred_log)
# accuracy LogisticRegression
print("LogisticRegression test Accuracy", accuracy_log * 100)


print('--------------------<DecisionTreeClassifier>--------------------------')
y_pred_tree = dtc_model.predict(X_data_test_fs)
accuracy_tree = accuracy_score(y_data_test_encod, y_pred_tree)
# accuracy DecisionTreeClassifier
print("DecisionTreeClassifier test Accuracy", accuracy_tree * 100)


print('--------------------<RandomForestClassifier>--------------------------')
y_pred_forest = rfc_model.predict(X_data_test_fs)
# accuracy RandomForestClassifier
accuracy_forest = accuracy_score(y_data_test_encod, y_pred_forest, normalize=True)
print("RandomForestClassifier test Accuracy", accuracy_forest * 100)


print('--------------------<AdaBoostClassifier>--------------------------')
predictions_ada = adaboost_classifier_model.predict(X_data_test_fs)
# accuracy AdaBoostClassifier
accuracy_ada = accuracy_score(y_data_test_encod, predictions_ada, normalize=True)
print("AdaBoostClassifier test Accuracy:", accuracy_ada * 100)

