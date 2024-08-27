

!pip install lucifer-ml
!pip install virtualenv
!pip install pandas_profiling
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from collections import Counter
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px
import pandas_profiling as pp
from luciferml.supervised import classification as cls ## Importing LuciferML
import virtualenv as virt

df = pd.read_csv("/content/customer  (1).csv")

df.head()

df.info()

df.isnull().sum()

df['SeniorCitizen'].fillna(df['SeniorCitizen'].mode()[0], inplace=True)
df['tenure'].fillna(df['tenure'].mode()[0], inplace=True)

df.isnull().sum()

df.isnull().sum()
df.head()

df.loc[df['Partner'] == 'Yes', 'Partner'] = '1'
df.loc[df['Partner'] == 'No', 'Partner'] = '0'

df.loc[df['Dependents'] == 'Yes', 'Dependents'] = '1'
df.loc[df['Dependents'] == 'No', 'Dependents'] = '0'

df.loc[df['PhoneService'] == 'Yes', 'PhoneService'] = '1'
df.loc[df['PhoneService'] == 'No', 'PhoneService'] = '0'

df.loc[df['PaperlessBilling'] == 'Yes', 'PaperlessBilling'] = '1'
df.loc[df['PaperlessBilling'] == 'No', 'PaperlessBilling'] = '0'

df.loc[df['Churn'] == 'Yes', 'Churn'] = '1'
df.loc[df['Churn'] == 'No', 'Churn'] = '0'



df['Partner']=df['Partner'].astype(float)
df['Churn']=df['Churn'].astype(float)
df['Dependents']=df['Dependents'].astype(float)
df['PhoneService']=df['PhoneService'].astype(float)
df['PaperlessBilling']=df['PaperlessBilling'].astype(float)

df.shape

df.describe().T.style.bar(
    subset=['mean'],
    color='Reds').background_gradient(
    subset=['std'], cmap='ocean').background_gradient(subset=['50%'], cmap='PuBu')

churn_count = df['Churn'].value_counts()
print(churn_count)
churn_count.plot.pie();

plt.figure(figsize=(6,5))
sns.countplot(x="Churn", data=df, palette='magma');

df.columns

def boxhistplot(columns,data):
    fig = px.histogram(df, x = df[column], color = 'Churn')
    fig.show()
    fig2 = px.box(df, x = df[column], color = 'Churn')
    fig2.show()

col = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
for column in col:
    boxhistplot(column,df)

sns.heatmap(df.corr(), annot=True, cmap="flag");

sns.pairplot(df, hue="Churn", palette="magma");

df.sample()


accuracy_scores = {}



df.sample()






df.head(10)

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def predict(X, y, user_data):
    regressor = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=68)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(user_data)
    accuracy = regressor.score(X_test, y_test)
    return predictions ,accuracy


X_columns = ['PhoneService', 'Partner', 'Dependents', 'SeniorCitizen', 'tenure', 'PaperlessBilling', 'MonthlyCharges']
X = df[X_columns] 
y = df['Churn'] 

user_input = {
    'PhoneService': [float(input("PhoneService: "))],
    'Partner': [float(input("Partner: "))],
    'Dependents': [float(input("Dependents: "))],
    'SeniorCitizen': [float(input("SeniorCitizen: "))],
    'tenure': [float(input("tenure: "))],
    'PaperlessBilling': [float(input("PaperlessBilling: "))],
    'MonthlyCharges': [float(input("MonthlyCharges: "))]
}

user_data = pd.DataFrame(user_input, columns=X_columns)  # Create a DataFrame from the user input

prediction,accuracy = predict(X, y, user_data)
print("Prediction:", prediction)
print("Accuracy:", accuracy)