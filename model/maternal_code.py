import pandas as pd

#loding the dataset
data = pd.DataFrame(pd.read_csv("model\Maternal Health Risk Data Set.csv"))

#Correlation of variables
import matplotlib.pyplot as plt
import seaborn as sns

y = data['RiskLevel']
X = data.drop(['RiskLevel'], axis=1)
# Splitting dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)

from sklearn.ensemble import RandomForestClassifier
rf_classifier= RandomForestClassifier(n_estimators=100, random_state=0)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

adaboost_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_leaf_nodes=8),
    n_estimators=300,
    algorithm="SAMME",
    random_state=42,
)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.25, max_features=2, max_depth=2, random_state=0)
model = VotingClassifier(estimators=[('rf', rf_classifier), ('adboost', adaboost_clf), ('gdboost', gb_clf)], voting='soft', weights=[2.5, 2, 1]).fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test,y_pred)*100)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
import pickle
with open('matRisk.pickle', 'wb') as f:
    pickle.dump(model, f)

# Save column names
import json
columns = {
    "data_columns": [col.lower() for col in X.columns]
}

with open('columns.json', 'w') as f:
    f.write(json.dumps(columns))