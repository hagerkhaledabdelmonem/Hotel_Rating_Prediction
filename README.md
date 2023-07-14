# Hotel_Rating_Prediction

There are 2 important question we are think about :
 - Can we make trips cozier by using data science?
 - Can we predict what score a reviewer will give a hotel using features about the hotel in combination with the reviewer history and each review’s language?

## Objective
- Apply different machine learning algorithms to real-world tasks.
- Increase knowledge about the workflow of the machine learning tasks.
- Learn how to clean data, applying pre-processing, feature engineering, regression, and classification methods. 
   
## Main Steps: :
- Apply pre-processing on the provided dataset. (Preprocess all the features even if we won’t use them later after feature selection)
- Apply Feature Selection and Experiment with regression techniques to reduce the error on prediction of the **Reviewer Score** 
- Drop a feature only after preprocessing and with valid reason)

## Preprocessing: 
- Before building models, We need to make sure that the dataset is clean and ready-to-use.

## Regression:
- Split your dataset into 80% training and 20% testing.
- Apply different regression techniques to find the model that fit data with minimum error.
  
## Classification: 
- Split your dataset into 80% training and 20% testing.
- Apply different models to classify each sample into distinct classes.
- Choose two hyperparameters to vary. Study at three different choices for each hyperparameter. When varying one hyperparameter, all the other hyperparameters should be fixed.


#### Regression Models Table :

|    Classification Models   |  Lasso_cv  |    Linear Regression     |       RandomForestRegressor           |          DecisionTree         |      Polynomial Regression      |  
|         :----:             |       :----:          |        :----:        |      :----:         |         :----:       |         :----:       |
|         R_square           | 0.38633161183320464   |  0.3863816691747861  |       0.4473832302938534         |           0.40971277175647103       |    0.4283665577931427  |
|        Mean Square Error      |  1.6544396419474405   |  1.65430468819737 |  1.4898455065250795 |  1.591404428471118   |     1.5411141353966     |








#### Classification Models Table :

|    Classification Models   |  LogisticRegression  |    Decision Tree     |       SVM           |          KNN         |  
|         :----:             |       :----:         |        :----:        |      :----:         |         :----:       |
|      Train Accuracy        |0.907035175879397   |  0.9371859296482412  |        0.9422110552763819         |           0.9221105527638191       |
|      Test Accuracy         |  0.9298245614035088   |  0.9181286549707602  |  0.8713450292397661 |  0.8888888888888888  |





  
# Collaborators:
- <a href="https://github.com/hagerkhaledabdelmonem">Hager Khaled</a><br>
- <a href="https://github.com/YasminHamada">Yasmin Hamada</a><br>
- <a href="https://github.com/Nourhan613">Nourhan Mohamed</a><br>
- <a href="https://github.com/nadakeshka">Nada Keshka</a><br>
- <a href="https://github.com/Basma-Ahmed24">Basma Ahmed</a><br>
