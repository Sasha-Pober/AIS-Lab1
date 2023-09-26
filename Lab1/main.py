import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('Iris.csv')

# Подивимось на перші 5 рядків датасету
print(df.head())

# Розділення даних на ознаки та цільову змінну
X = df.drop(['Id', 'Species'], axis=1)
y = df['Species']

# Розділення даних на тренувальний та тестовий датасети
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Ініціалізація та навчання моделі
clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

test_data = np.array([[4.9, 3.1, 1.5, 0.1]])

prediction = clf.predict(test_data)
print("Prediction: ", prediction)
print("Accuracy:", accuracy_score(y_test, y_pred))

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)
plt.show()

