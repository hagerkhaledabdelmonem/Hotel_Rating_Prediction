import pickle
from joblib import Memory
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoCV
from sklearn import metrics
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import Preprocessing as pp
import Visualization
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv('hotel-regression-dataset.csv', parse_dates=['Review_Date'], dayfirst=True)

# Visualization Plots
Visualization.Plots(data)


# for Pickels
models = {}


# Split data to X & Y
X = data.drop(columns=['Reviewer_Score'], axis=1)  # Features
Y = data['Reviewer_Score']  # Label

# Split to Train & Test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=10)

pp.nulls_X(X_train)
X_train , X_test = pp.fill_null(X_train ,X_test)

# Preprocessing
X_train = pp.preprocessing(X_train)
X_test = pp.preprocessing(X_test)

# Feature_Encoding
cols = X_train.select_dtypes(include=["object"]).columns.tolist()
X_train ,X_test = pp.Feature_Encoder(X_train, X_test,cols, models)

# access the LabelEncoder(X_Data)
lbl_encoder = models['label encoder']

dir = 'preprocessing_reg_file'
mem = Memory(dir)
pre_reg = mem.cache(pp.preprocessing)

X_train, X_test = pp.scaler(X_train, X_test,models)


X_train_fs, X_test_fs = pp.select_features(X_train, y_train, X_test,models)

print(" --------------------Lasso_CV----------------")
# Lasso Cross validation
lasso_cv = LassoCV(alphas=[0.00001])
lasso_cv.fit(X_train_fs, y_train)
models['Lasso_CV'] = lasso_cv
ypred_test_lasso_cv = lasso_cv.predict(X_test_fs)
# Model R2 Score & MSE
print("Model_Lasso_CV R2 Score: ", r2_score(y_test, ypred_test_lasso_cv))
print('Mean Square Error', metrics.mean_squared_error(y_test, ypred_test_lasso_cv))


print("--------------------linear----------------")
cls = linear_model.LinearRegression()
cls.fit(X_train_fs, y_train)
models['linear'] = cls
y_test_predicted = cls.predict(X_test_fs)

# plot
plt.scatter(y_test, y_test_predicted)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='dashed')
plt.title('Linear Regression Model')
plt.xlabel('Actual Data')
plt.ylabel('Prediction Data')
plt.show()

print("Model_Poly r2 Score: ", r2_score(y_test, y_test_predicted))
print('Mean Square Error linear: ', metrics.mean_squared_error(y_test, y_test_predicted))


print("-----------------RandomForestRegressor-------------------")
rf = RandomForestRegressor(max_depth=10, random_state=2)
rf.fit(X_train_fs, y_train)
models['RandomForestRegressor'] = rf
rf_pred = rf.predict(X_test_fs)
# Model R2 Score & MSE
print("Model RandomForestRegressor R2 Score: ", r2_score(y_test, rf_pred))
print('Mean Square Error RandomForestRegressor: ', metrics.mean_squared_error(y_test, rf_pred))


print("--------------------DecisionTreeRegressor----------------")
regr_2 = DecisionTreeRegressor(random_state=10, max_depth=7, min_samples_leaf=5)
regr_2.fit(X_train_fs, y_train)
models['DecisionTreeRegressor'] = regr_2
y_2 = regr_2.predict(X_test_fs)
# Model R2 Score & MSE
print("Model DecisionTreeRegressor R2 Score: ", r2_score(y_test, y_2))
print('Mean Square Error DecisionTreeRegressor: ', metrics.mean_squared_error(y_test, y_2))


print(" --------------------Polynomial----------------")
# create the model object
poly_features = PolynomialFeatures(degree=3)
# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train_fs)
models['Polynomial'] = poly_features
# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)
models['Polynomial_linear'] = poly_model
prediction = poly_model.predict(poly_features.fit_transform(X_test_fs))
# Model R2 Score & MSE
print("Model Polynomial R2 Score: ", r2_score(y_test, prediction))
print('Mean Square Error Polynomial: ', metrics.mean_squared_error(y_test, prediction))


# Use the saved models By pickle
with open("reg_hotel_models.pkl", "wb") as f:
    pickle.dump(models, f)

with open("reg_hotel_models.pkl", "rb") as f:
     models = pickle.load(f)



# access the lasso_cv model
lasso_cv_model = models['Lasso_CV']
# access the DecisionTreeRegressor model
dtr_model = models['DecisionTreeRegressor']
# access the RandomForestRegressor model
rfr_model = models['RandomForestRegressor']
# access the linear model
linear_model = models['linear']
# access the Polynomial model
Polynomial_model =models['Polynomial']
Polynomial_model_Linear=models['Polynomial_linear']

# access the Standard Scaler
scalerx_train = models['Standard Scaler']
# access the SelectKBest
fs = models['SelectKBest']

datafile = input("Enter the test data : " )
data_test = pd.read_csv(datafile + '.csv', parse_dates=['Review_Date'], dayfirst=True)
x_data_test =data_test.drop(columns=['Reviewer_Score'], axis=1)  # Features
y_data_test = data_test['Reviewer_Score']


# in file test data
with open('reg_column_info.plk', 'rb') as sa:
    column_info = pickle.load(sa)

for col in x_data_test.columns:
    if (x_data_test[col].isnull().sum() != 0):
        if column_info[col]['type'] == 'categorical':
            x_data_test[col] = x_data_test[col].fillna(column_info[col]['value'])
        else:
            x_data_test[col] = x_data_test[col].fillna(column_info[col]['value'])

x_data_test=pp.preprocessing(x_data_test)


# this code in the test data after X_test_data preprocessing
cols = x_data_test.select_dtypes(include=["object"]).columns.tolist()
for col in cols:
    encoding_dict = {label: index for index, label in enumerate(lbl_encoder.classes_)}
    unknown_val = len(encoding_dict) # Assign a value to represent "unknown"
    x_data_test[col] = [encoding_dict.get(label, unknown_val) for label in x_data_test[col]]


X_test_scaled = scalerx_train.transform(x_data_test)
X_data_test_fs = fs.transform(X_test_scaled)



print('--------------------<Lasso_CV>--------------------------')
y_pred_lasso = lasso_cv_model.predict(X_data_test_fs)
accuracy_log = r2_score(y_data_test, y_pred_lasso)
# r2score Lasso_CV
print("Lasso_CV test Accuracy", accuracy_log * 100)


print('--------------------<DecisionTreeRegressor >--------------------------')
y_pred_tree = dtr_model .predict(X_data_test_fs)
accuracy_tree = r2_score(y_data_test, y_pred_tree)
# r2score DecisionTreeRegressor
print("DecisionTreeRegressor test Accuracy", accuracy_tree * 100)


print('--------------------<RandomForestRegressor>--------------------------')
y_pred_forest = rfr_model.predict(X_data_test_fs)
# r2score RandomForestRegressor
accuracy_forest = r2_score(y_data_test, y_pred_forest)
print("RandomForestRegressor test Accuracy", accuracy_forest * 100)


print('--------------------<linear>--------------------------')
y_pred_linear = linear_model.predict(X_data_test_fs)
# r2score linear
accuracy_linear = r2_score(y_data_test, y_pred_linear)
print("linear test Accuracy", accuracy_linear * 100)


print('--------------------<Polynomial>--------------------------')
# fit the transformed features to Linear Regression
predictions_Polynomial = Polynomial_model_Linear.predict(Polynomial_model.transform(X_data_test_fs))
# r2score AdaBoostClassifier
accuracy_Polynomial = r2_score(y_data_test, predictions_Polynomial)
print("AdaBoostClassifier test Accuracy:", accuracy_Polynomial * 100)

