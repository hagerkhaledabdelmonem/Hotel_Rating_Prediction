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
- Apply sentiment analysis using the review column

## Preprocessing: 
- Before building models, We need to make sure that the dataset is clean and ready-to-use.

## Regression:
- Split your dataset into 80% training and 20% testing.
- Apply different regression techniques to find the model that fit data with minimum error.
  
## Classification: 
- Split your dataset into 80% training and 20% testing.
- Apply different models to classify each sample into distinct classes.
- Choose two hyperparameters to vary. Study at three different choices for each hyperparameter. When varying one hyperparameter, all the other hyperparameters should be fixed.


### Regression Models Table :

|    Regression Models   |  Lasso_cv  |    Linear Regression     |       RandomForestRegressor           |          DecisionTree         |      Polynomial Regression      |  
|         :----:             |       :----:          |        :----:        |      :----:         |         :----:       |         :----:       |
|         R_square           |      0.3849440376625387   |  0.38494403287406764  |      0.45004317773035707         |   0.41176508418282953       |     0.4341312285364416 |
|        Mean Square Error      |     1.658180518548482   |  1.6581805314581195 |  1.482674333022735 |  1.5858714287251607   |     1.5255726801344527     |


#### The best models when we use split data train , validation and test are RandomForestRegressor , DecisionTreeRegressor and Linear or Lasso_CV because they have high R_square score and low mean square error


### Classification Models Table :

|    Classification Models   |    LogisticRegression   |    Decision Tree     |       RandomForestClassifier   |        AdaBoostClassifier      |  
|         :----:             |       :----:            |        :----:        |            :----:              |              :----:            |
|       Accuracy             |    71.24330468628904    |  71.39314193203934   |        72.0321030604688        |        72.18538484060417       |


#### The best models for Classificatiion when we use split data train and test are RandomForestRegressor and AdaBoostClassifier, they have high Accuracy 


  
# Collaborators:
- <a href="https://github.com/hagerkhaledabdelmonem">Hager Khaled</a><br>
- <a href="https://github.com/YasminHamada">Yasmin Hamada</a><br>
- <a href="https://github.com/Nourhan613">Nourhan Mohamed</a><br>
- <a href="https://github.com/nadakeshka">Nada Keshka</a><br>
- <a href="https://github.com/Basma-Ahmed24">Basma Ahmed</a><br>
