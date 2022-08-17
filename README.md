# Hamoye-Stage-B
import pandas as pd

#Convert the dictionary into DataFrame
df = pd.DataFrame(df)
  
# Remove column name 'data' and 'lights'
import numpy as np
df=pd.read_csv('C:\\Users\\D-IKE\\Desktop\\HamoyeCODES\\energydata_complete.csv')
df.drop(['date', 'lights'], axis=1)

#normalising the dataset to a common scale using min max scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalised_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
features_df = normalised_df.drop(columns=['Appliances'])
heating_target = normalised_df['Appliances']

#splitting dataset into training and testing dataset.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features_df, heating_target,
test_size=70, random_state=42)

from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
#fit the model to the training dataset
linear_model.fit(x_train, y_train)
#obtain predictions
predicted_values = linear_model.predict(x_test)
