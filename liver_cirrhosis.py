import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier


#Path
cirrhosis_dataset = 'C:/Users/ASUS/Downloads/cirrhosis_new.csv'

dataf = pd.read_csv(cirrhosis_dataset)
print(dataf.head())

Null_values = dataf.isnull()

plt.figure(figsize=(10, 6))
sns.heatmap(Null_values, cmap='viridis', cbar=False)
plt.title('Missing Values in Dataset')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.show()


dataf['Age'] = dataf['Age'] / 365.25
dataf['Age'] = dataf['Age'].apply(math.floor)
dataf.fillna('NA', inplace=True)

print(dataf['Age'].head())
dataf.to_csv('processed_data.csv', index=False)

#processed data path
processed_data = 'C:/Users/ASUS/Downloads/processed_data.csv'
dataf1 = pd.read_csv(processed_data)

print(dataf1.head())

Null_values1 = dataf1.isnull()

plt.figure(figsize=(10, 6))
sns.heatmap(Null_values1, cmap='viridis', cbar=False)
plt.title('Missing Values in Dataset')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.show()

dataf1.info()
dataf1.head()
dataf1.isnull().sum()
dataf1.duplicated().sum()

dataf1.drop(['ID',"N_Days","Status"], axis=1, inplace=True)

fig = plt.figure(figsize=(5,5))
sns.countplot(x='Stage', data =dataf1)
plt.show()


X= dataf1.drop('Stage', axis=1)
y = dataf1['Stage']

random = RandomOverSampler(random_state=42)
X_resampled, y_resampled = random.fit_resample(X, y)
dataf1 = pd.concat([X_resampled, y_resampled], axis=1)
dataf1 = dataf1.sample(frac=1, random_state=42).reset_index(drop=True)

fig = plt.figure(figsize=(5,5))
sns.countplot(x='Stage', data =df1)
plt.show()

category_columns = ['Stage', 'Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
for column in category_columns:
    print(f"Column: {column}")
    print(dataf1[column].value_counts())
    print()


def columnvaluechange(dataf1):
    dataf1['Sex'] = dataf1['Sex'].replace(['F', 'M'], [0, 1])
    dataf1['Sex'] = pd.to_numeric(dataf1['Sex'])

    dataf1['Ascites'] = dataf1['Ascites'].replace(['N', 'Y'], [0, 1])
    dataf1['Ascites'] = pd.to_numeric(dataf1['Ascites'])

    dataf1['Hepatomegaly'] = dataf1['Hepatomegaly'].replace(['N', 'Y'], [0, 1])
    dataf1['Hepatomegaly'] = pd.to_numeric(dataf1['Hepatomegaly'])

    dataf1['Spiders'] = dataf1['Spiders'].replace(['N', 'Y'], [0, 1])
    dataf1['Spiders'] = pd.to_numeric(dataf1['Spiders'])

    dataf1['Edema'] = dataf1['Edema'].replace(['N', 'S', 'Y'], [0.2, 0.4, 0.6])
    dataf1['Edema'] = pd.to_numeric(dataf1['Edema'])

    dataf1['Drug'] = dataf1['Drug'].replace(['D-penicillamine', 'Placebo'], [0, 1])
    dataf1['Drug'] = pd.to_numeric(dataf1['Drug'])
    return df1

dataf1 = columnvaluechange(dataf1)

categorical_columns = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
for column in categorical_columns:
    print(f"Column: {column}")
    print(df1[column].value_counts())
    print()

dataf1.head(15)

correlation_matrix = dataf1.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.show()

num_columns = [ 'Age', 'Bilirubin', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Platelets', 'Prothrombin']
for column in num_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(dataf1[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.show()

sns.countplot(x='Stage', hue='Sex', data=dataf1)
plt.title("Count of Stage based on Sex")
plt.show()

label_encoder = LabelEncoder()
for column in dataf1:
    dataf1[column] = label_encoder.fit_transform(dataf1[column])

dataf1 = dataf1.dropna(subset=['Drug'])

auto_fill = ['Bilirubin', 'Copper', 'Alk_Phos', 'SGOT', 'Platelets', 'Prothrombin', 'Stage']
dataf1[auto_fill] = dataf1[auto_fill].fillna(dataf1[auto_fill].mean())

dataf1.info()

Null_values2 = dataf1.isnull()

plt.figure(figsize=(10, 6))
sns.heatmap(missing_values, cmap='viridis', cbar=False)
plt.title('Missing Values in Dataset')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.show()

X = dataf1.drop(columns=['Stage'])
y = dataf1['Stage']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,shuffle=True)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train[:5])
print(X_test[:5])

xgboost = XGBClassifier()
xgboost.fit(X_train, y_train)
print(xgboost.score(X_train, y_train))
y_pred1 = xgboost.predict(X_test)

xgb_predictions = xgboost.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
print(f'XGB Accuracy: {xgb_accuracy:.2f}')

cm = confusion_matrix(y_test,xgb_predictions)
print(cm)
plt.figure(figsize=(7, 6))
sns.heatmap(confusion_matrix(y_test, xgb_predictions), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for XG Boosting')
plt.show()

randomf=RandomForestClassifier()
randomf.fit(X_train,y_train)

print(randomf.score(X_train, y_train))
y_pred1 = randomf.predict(X_test)

prediction3=randomf.predict(X_test)
randomf_acc = accuracy_score(y_test, prediction3)
print(f'Random_Forest_accuracy: {randomf_acc:.2f}')

cm1 = confusion_matrix(y_test, prediction3)
print(cm1)
plt.figure(figsize=(7, 6))
sns.heatmap(confusion_matrix(y_test, prediction3), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for RandomForestClassifier')
plt.show()

supportvm_classifier = SVC(kernel='rbf')
supportvm_classifier.fit(X_train, y_train)

print(supportvm_classifier.score(X_train, y_train))
y_pred1 = supportvm_classifier.predict(X_test)

supportvm_predictions = supportvm_classifier.predict(X_test)
supportvm_accuracy = accuracy_score(y_test, supportvm_predictions)
print(f'SVM Accuracy: {supportvm_accuracy:.2f}')

cm2 = confusion_matrix(y_test, supportvm_predictions)
print(cm2)
plt.figure(figsize=(7, 6))
sns.heatmap(confusion_matrix(y_test, supportvm_predictions), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for SVM')
plt.show()
