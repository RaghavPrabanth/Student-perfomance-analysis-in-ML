import numpy as np
import pandas as pd

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import dabl

data=pd.read_csv(r"C:\Users\Raghav Prabanth\OneDrive\Desktop\python learner files\StudentsPerformance.csv")

# getting the shape of the data
print(data.shape)

no_of_columns = data.shape[0]
percentage_of_missing_data = data.isnull().sum()/no_of_columns
print(percentage_of_missing_data)

plt.rcParams['figure.figsize'] = (18, 6)
plt.style.use('fivethirtyeight')
dabl.plot(data, target_col='math score')
dabl.plot(data, target_col='reading score')

from math import ceil
import warnings
warnings.filterwarnings('ignore')

data['total_score'] = data['math score'] + data['reading score'] + data['writing score']
data['percentage'] = data['total_score'] / 3

for i in range(len(data)):
    data.loc[i, 'percentage'] = ceil(data.loc[i, 'percentage'])

plt.rcParams['figure.figsize'] = (15, 9)
sns.distplot(data['percentage'], color='orange')
plt.title('Comparison of percentage scored by all the students', fontweight=30, fontsize=20)
plt.xlabel('Percentage scored')
plt.ylabel('Count')
plt.show()

sns.distplot(data['total_score'], color='magenta')
plt.title('Comparison of total score of all the students', fontweight=30, fontsize=20)
plt.xlabel('Total score scored by the students')
plt.ylabel('Count')
plt.show()

def getgrade(percentage):
    if percentage >= 90:
        return 'O'
    elif percentage >= 80:
        return 'A'
    elif percentage >= 70:
        return 'B'
    elif percentage >= 60:
        return 'C'
    elif percentage >= 40:
        return 'D'
    else:
        return 'E'

data['grades'] = data['percentage'].apply(getgrade)
print(data['grades'].value_counts())

from sklearn.preprocessing import LabelEncoder

# creating an encoder
le = LabelEncoder()

# label encoding for test preparation course
data['test preparation course'] = le.fit_transform(data['test preparation course'])

# label encoding for lunch
data['lunch'] = le.fit_transform(data['lunch'])

# label encoding for race/ethnicity
data['race/ethnicity'] = data['race/ethnicity'].map({
    'group A': 1,
    'group B': 2,
    'group C': 3,
    'group D': 4,
    'group E': 5
})

# label encoding for parental level of education
data['parental level of education'] = le.fit_transform(data['parental level of education'])

# label encoding for gender
data['gender'] = le.fit_transform(data['gender'])

# Feature matrix (X) and target variable (y)
x = data[['gender', 'race/ethnicity', 'parental level of education', 
          'lunch', 'test preparation course', 'math score', 'reading score', 'writing score', 
          'total_score', 'percentage']]
y = data['grades']

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=45)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# importing the MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# creating a scaler
mm = MinMaxScaler()

# scaling the independent variables
x_train = mm.fit_transform(x_train)
x_test = mm.transform(x_test)

from sklearn.decomposition import PCA

# creating a principal component analysis model
pca = PCA(n_components=None)

# fitting PCA
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
print(explained_variance)

# reducing to 2 principal components
pca = PCA(n_components=2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

from sklearn.ensemble import RandomForestClassifier

# creating a model
model = RandomForestClassifier()

# training the model
model.fit(x_train, y_train)

# predicting on the test data
y_pred = model.predict(x_test)

# calculating accuracies
print("Training Accuracy:", model.score(x_train, y_train))
print("Testing Accuracy:", model.score(x_test, y_test))

from sklearn.metrics import confusion_matrix

# confusion matrix
cm = confusion_matrix(y_test, y_pred)

# plot the confusion matrix
plt.rcParams['figure.figsize'] = (8, 8)
sns.heatmap(cm, annot=True, cmap='Reds')
plt.title('Confusion Matrix for Random Forest', fontweight=30, fontsize=20)
plt.show()

from pandas.plotting import radviz
fig, ax = plt.subplots(figsize=(12, 12))
new_df = pd.concat([pd.DataFrame(x, columns=['PC1', 'PC2']), y], axis=1)
radviz(new_df, 'grades', ax=ax, colormap="rocket")
plt.title('Radial Visualization for Target', fontsize=20)
plt.show()
