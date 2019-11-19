# -*- coding: utf-8 -*-

"""

	Very first steps in ML.
	Data Preprocessing

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#Reading dataset with pandas
dataset = pd.read_csv('Data.csv')

#The last column is the Y values so we should not take the last column in our X datas. 
X = dataset.iloc[:,:-1].values
#Labels as we mentioned above.
y = dataset.iloc[:,3].values

# In some cases dataset may contain null values and this values could be a worst situation in our models. So we have to get rid of these values or fill its.

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#OneHotEncoder and LabelEncoder use for the encode to your text formatted datas.

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
label_encoder_X = LabelEncoder()
X[:,0] = label_encoder_X.fit_transform(X[:,0])
one_hot_encoder = OneHotEncoder(categorical_features = [0])
X = one_hot_encoder.fit_transform(X).toarray()

label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

#Train and test splitting
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Scalers might be good choose of different scale of data, I mean for example if one values goes to 1000 an other goes to just 0.1, so it's not good for the model.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
