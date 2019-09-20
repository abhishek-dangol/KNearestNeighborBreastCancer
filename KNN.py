import numpy as np
from sklearn import preprocessing, model_selection, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.csv')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

###drop the clas coumn
X = np.array(df.drop(['class'], 1))

###this is what we are predicting
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.2)

# defining the classifier
clf = neighbors.KNeighborsClassifier()

clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

# random values for testing
example_measures = np.array(
    [[4, 2, 1, 1, 1, 2, 3, 2, 1], [8, 7, 5, 10, 10, 10, 1, 4, 8]])
example_measures = example_measures.reshape(len(example_measures), -1)

prediction = clf.predict(example_measures)

###2 is for benign and 4 is for malignant
print(prediction)
