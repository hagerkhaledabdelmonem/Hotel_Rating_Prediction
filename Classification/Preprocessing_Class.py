from ast import literal_eval
import pickle
from textblob import TextBlob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

def nulls_X(X):
    # file preprocessing
    # save the column information
    modes = {}
    means = {}
    for col in X.columns:
        if X[col].dtypes == 'O':
            modes[col] = X[col].mode()[0]
        else:
            means[col] = X[col].mean()

    column_info = {}
    for col in X.columns:
        if col in modes.keys():
            column_info[col] = {'type': 'categorical', 'value': modes[col]}
        else:
            column_info[col] = {'type': 'numerical', 'value': means[col]}

    with open('column_info.plk', 'wb') as sa:
        pickle.dump(column_info, sa)


# function fill_null
def fill_null(X_train, X_test):
    with open('column_info.plk', 'rb') as sa:
        column_info = pickle.load(sa)
    for col in X_train.columns:
        if (X_train[col].isnull().sum() != 0):
            X_train[col] = X_train[col].fillna(column_info[col]['value'])
        else:
            X_train[col] = X_train[col].fillna(column_info[col]['value'])
    for col in X_test.columns:
        if (X_test[col].isnull().sum() != 0):
            if column_info[col]['type'] == 'categorical':
                X_test[col] = X_test[col].fillna(column_info[col]['value'])
            else:
                X_test[col] = X_test[col].fillna(column_info[col]['value'])

    return X_train, X_test


def scaler(x_train, x_text,models):
    lx = x_train.columns
    ltest = x_text.columns
    scaler = StandardScaler()
    scaler.fit(x_train)
    models['Standard Scaler']=scaler
    x_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_text)
    final_x = pd.DataFrame(x_scaled, columns=lx)
    final_x_test = pd.DataFrame(x_test_scaled, columns=ltest)
    return final_x, final_x_test



def Feature_Encoder(X_train,X_test ,cols, models):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X_train[c].values))
        models['label encoder'] = lbl
        encoding_dict = {label: index for index, label in enumerate(lbl.classes_)}
        unknown_val = len(encoding_dict) # Assign a value to represent "unknown"
        X_train[c] = lbl.transform(list(X_train[c].values))
        X_test[c] = [encoding_dict.get(label, unknown_val) for label in X_test[c]]
    return X_train, X_test

def impute(column):
    column = column[0]
    if (type(column) != list):
        return "".join(literal_eval(column))
    else:
        return column


def Polarity_Positive(x):
    if (x >= -1 and x <= 0):
        return 'No Positive'
    if (x > 0 and x <= 1):
        return 'Positive'


def Polarity_Negative(x):
    if (x >= -1 and x < 0):
        return 'Negative'
    if (x >= 0 and x <= 1):
        return 'No Negative'



#function of feature selection
def select_features(X_train, y_train, X_test,models):
        # configure to select all features
        fs = SelectKBest(score_func=f_classif)
        # learn relationship from training data
        fs.fit(X_train, y_train)
        models['SelectKBest']=fs
        # transform train input data
        X_train_fs = fs.transform(X_train)
        # transform test input data
        X_test_fs = fs.transform(X_test)
        return X_train_fs, X_test_fs


def preprocessing(data):
    # Number of nulls of each column
    print(data.isna().sum())
    # Information about the DataFrame
    print(data.info())

    data['Review_Date'] = pd.to_datetime(data['Review_Date'])
    data.insert(3, "Day", data['Review_Date'].dt.day, True)
    data.insert(4, "Month", data['Review_Date'].dt.month, True)
    data.insert(5, "Year", data['Review_Date'].dt.year, True)
    data.drop(['Review_Date'], axis=1, inplace=True)

    # Days_since_review
    data['days_since_review'] = data['days_since_review'].str.replace('[days]', ' ').astype(int)

    # Hotel_Address
    data["Country"] = data.Hotel_Address.apply(lambda x: x.split(' ')[-1])
    data.drop(["Hotel_Address"], axis=1, inplace=True)

    # Tags column
    data["Tags"] = data[["Tags"]].apply(impute, axis=1)
    data["Tags"] = data["Tags"].str.lower()
    data['Trip Type'] = data['Tags'].str.extract('(leisure trip|business trip)', expand=False)
    data['booking method'] = data['Tags'].str.extract('(submitted from a mobile device)', expand=False)
    data['type of people'] = data['Tags'].str.extract(
        '(family with older children|couple|group|solo traveler|family with young children|travelers with friends)',
        expand=False)
    data['Nights Stayed'] = data['Tags'].str.extract('(\d+) nights?', expand=False)
    data['Room Type'] = data['Tags'].str.replace(
        '(leisure trip|business trip|submitted from a mobile device|family with older children|'
        'couple|group|solo traveler|family with young children|travelers with friends|(\d+) nights?|stayed) ', '',
        regex=True)

    data.drop(["Tags"], axis=1, inplace=True)

    tags_nan_cols = ('Trip Type', 'Nights Stayed')
    for col in tags_nan_cols:
        data[col] = data[col].fillna(data[col].mode()[0])

    data['booking method'] = data['booking method'].fillna('Unknown')

    # Negative_Review & Positive_Review
    data['Positive_Sentiment'] = data['Positive_Review'].apply(lambda x: (TextBlob(x).sentiment.polarity))
    data['Positive_Sentiment'] = data['Positive_Sentiment'].apply(lambda x: Polarity_Positive(x))

    data['Negative_Sentiment'] = data['Negative_Review'].apply(lambda x: (TextBlob(x).sentiment.polarity))
    data['Negative_Sentiment'] = data['Negative_Sentiment'].apply(lambda x: Polarity_Negative(x))

    data.drop(['Negative_Review', 'Positive_Review'], axis=1, inplace=True)


    return data
