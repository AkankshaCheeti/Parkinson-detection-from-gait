import os
import pandas as pd
import csv
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Train two Random Forest models
warnings.filterwarnings('ignore') 


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load and prepare the datasets
data = pd.read_csv("../dataset_tl/Parkinson_CV.tab", delimiter="\t")
print(data)


# Filter the last column as the label
label = data.iloc[:, -1]

label = label.replace({'Co': 0, 'Pt': 1})

# Remove the last column from the datasets
data = data.iloc[:, :-1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)


#Ensemble model
# Instantiate the models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)

# Train the models
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
ada_model.fit(X_train, y_train)

# Make predictions on the test set
rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)
ada_pred = ada_model.predict(X_test)

# Combine the predictions using a simple majority vote
ensemble_pred = np.round((rf_pred + gb_pred + ada_pred) / 3)

# Evaluate the ensemble's accuracy
ensemble_acc = accuracy_score(y_test, ensemble_pred)
print("Ensemble accuracy:", ensemble_acc)

#transfer learning
print('***** Transfer Learning ***')

test = pd.read_csv("../dataset_tl/Parkinson_CV_dataset2_transformed.tab", delimiter="\t")

# Filter the last column as the label
label = test.iloc[:, -1]

label = label.replace({'Co': 0, 'Pt': 1})
# Remove the last column from the datasets
test = test.iloc[:, :-1]
# Make predictions on the test set
rf_pred = rf_model.predict(test)
gb_pred = gb_model.predict(test)
ada_pred = ada_model.predict(test)

# Combine the predictions using a simple majority vote
ensemble_pred = np.round((rf_pred + gb_pred + ada_pred) / 3)

# Evaluate the ensemble's accuracy
ensemble_acc = accuracy_score(label, ensemble_pred)
print("Ensemble accuracy:", ensemble_acc)
