# deep-learning-challenge
## Overview
Alphabet Soup, a nonprofit foundation, requires a predictive tool to assist in the selection of funding recipients who are most likely to succeed in their ventures. My task is to utilize machine learning and neural networks to create a binary classifier using the available dataset. The dataset consists of over 34,000 organizations that have previously received funding from Alphabet Soup. Each organization's metadata, including various columns such as: 
- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special considerations for application
- ASK_AMT—Funding amount requested
- IS_SUCCESSFUL—Was the money used effectively
- 
## Data Preprocessing
Using Google Colab, I read in the charity_data.csv to a Pandas DataFrame. I determined  'IS_SUCCESSFUL' for my Target variable and all other columns for my features variable. I dropped `EIN`or `NAME` columns from  my model since they were not going to be used. I then used `pd.get_dummies()` to encode categorical variables and split the preprocessed data into a features array, `X`, and a target array,`y`. Using the `train_test_split` function, I split the data into training and testing datasets. To ensure that data is not heavily weighted in any one area, I scaled the training and testing feature datasets by creating a `StandardScaler` instance, fitting it to the training data, then used the `transform` function.
  
## Compile, Train and Evaluate the Model
After processing the data I created a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras. I created the first hidden layer, added a second hidden layer and an output layer.
<img width="500" src="https://github.com/nancygmz/deep-learning-challenge/blob/main/images/Screenshot%202023-06-07%20193430.png">

## Summary
The model was only able to achieve 72-73% accuracy, which is relatively low for a production model. Further optimization could be conducted to increase the hidden layers, neurons, and adding more epochs as there are many different applications, categories that we may need more neurons to make connections and longer run times to get above the 75% accuracy score.

