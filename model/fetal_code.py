# import numpy as np 
import pandas as pd
# from scipy import stats
# from sklearn import preprocessing
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.impute import KNNImputer
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv(r'model\fetal_health.csv')
print(df['fetal_health'].value_counts())
# df.info()

# df = df.drop_duplicates()

# fig, axes = plt.subplots(6, 4, figsize=(10, 12))
# axes = axes.flatten()
# # Create a boxplot for each continuous variable
# for i, column in enumerate(df.columns):
#     sns.boxplot(data=df, x=column, ax=axes[i])
#     axes[i].set_title(f"{column}", fontsize=10)
#     axes[i].set_xlabel("")  # Remove x-axis title

# # Remove empty subplot
# for i in range(len(df.columns), len(axes)):
#     fig.delaxes(axes[i])

# # Adjust layout
# plt.tight_layout()
# plt.show()

# df['fetal_health'] = df['fetal_health'].astype(int)

# df_features = df.drop(['fetal_health', 'histogram_number_of_zeroes'], axis=1)
# df_target = df['fetal_health']

# # split df at 70-30 ratio
# X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.3, random_state=123)

# scaler = MinMaxScaler()

# # Fit and transform the train_set & test_set, features only
# scaled_X_train = scaler.fit_transform(X_train)
# scaled_X_test = scaler.fit_transform(X_test)

# scaled_X_train = pd.DataFrame(scaled_X_train, columns = X_train.columns, index=X_train.index)
# scaled_X_train.head()

# scaled_X_test = pd.DataFrame(scaled_X_test, columns = X_test.columns, index=X_test.index)
# scaled_X_test.head()

# #NAIVE BAYES
# naive_bayes = MultinomialNB()

# # Train the NB model on the train data
# naive_bayes.fit(scaled_X_train, y_train)

# # Predict on test data
# y_pred_nb = naive_bayes.predict(scaled_X_test)

# # Calculate evaluation metrics
# accuracy_nb = accuracy_score(y_test, y_pred_nb)
# precision_nb = precision_score(y_test, y_pred_nb, average='weighted')
# recall_nb = recall_score(y_test, y_pred_nb, average='weighted')
# f1_nb = f1_score(y_test, y_pred_nb, average='weighted')

# # Print results
# print("Accuracy:", round(accuracy_nb, 3))
# print("Precision:", round(precision_nb, 3))
# print("Recall:", round(recall_nb, 3))
# print("F1-Score:", round(f1_nb, 3))

# #LOGISTIC REGRESSION
# log_reg = LogisticRegression(random_state=123)

# # Train the LR model on the train data
# log_reg.fit(scaled_X_train, y_train)

# # Predict on test data
# y_pred_lr = log_reg.predict(scaled_X_test)

# # Calculate evaluation metrics
# accuracy_lr = accuracy_score(y_test, y_pred_lr)
# precision_lr = precision_score(y_test, y_pred_lr, average='weighted')
# recall_lr = recall_score(y_test, y_pred_lr, average='weighted')
# f1_lr = f1_score(y_test, y_pred_lr, average='weighted')

# # Print results
# print("Accuracy:", round(accuracy_lr, 3))
# print("Precision:", round(precision_lr, 3))
# print("Recall:", round(recall_lr, 3))
# print("F1-Score:", round(f1_lr, 3))

# #DECISION TREE
# d_tree = DecisionTreeClassifier(random_state=123)

# # Train the DT model on the train data
# d_tree.fit(scaled_X_train, y_train)

# # Predict on test data
# y_pred_dt = d_tree.predict(scaled_X_test)

# # Calculate evaluation metrics
# accuracy_dt = accuracy_score(y_test, y_pred_dt)
# precision_dt = precision_score(y_test, y_pred_dt, average='weighted')
# recall_dt = recall_score(y_test, y_pred_dt, average='weighted')
# f1_dt = f1_score(y_test, y_pred_dt, average='weighted')

# # Print results
# print("Accuracy:", round(accuracy_dt, 3))
# print("Precision:", round(precision_dt, 3))
# print("Recall:", round(recall_dt, 3))
# print("F1-Score:", round(f1_dt, 3))

# #RANDOM FOREST
# r_forest = RandomForestClassifier(random_state=123)

# # Train the DT model on the train data
# r_forest.fit(scaled_X_train, y_train)

# # Predict on test data
# y_pred_rf = r_forest.predict(scaled_X_test)

# # Calculate evaluation metrics
# accuracy_rf = accuracy_score(y_test, y_pred_rf)
# precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
# recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
# f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

# # Print results
# print("Accuracy:", round(accuracy_rf, 3))
# print("Precision:", round(precision_rf, 3))
# print("Recall:", round(recall_rf, 3))
# print("F1-Score:", round(f1_rf, 3))

# #KNN
# knn = KNeighborsClassifier()

# # Train KNN on the train data
# knn.fit(scaled_X_train, y_train)

# # Predict on test data
# y_pred_knn = knn.predict(scaled_X_test)

# # Calculate evaluation metrics
# accuracy_knn = accuracy_score(y_test, y_pred_knn)
# precision_knn = precision_score(y_test, y_pred_knn, average='weighted')
# recall_knn = recall_score(y_test, y_pred_knn, average='weighted')
# f1_knn = f1_score(y_test, y_pred_knn, average='weighted')

# # Print results
# print("Accuracy:", round(accuracy_knn, 3))
# print("Precision:", round(precision_knn, 3))
# print("Recall:", round(recall_knn, 3))
# print("F1-Score:", round(f1_knn, 3))

# #SVM
# svm = SVC(random_state=123)

# # Train SVC on train data
# svm.fit(scaled_X_train, y_train)

# # Predict on the test data
# y_pred_svm = svm.predict(scaled_X_test)

# # Calculate evaluation metrics
# accuracy_svm = accuracy_score(y_test, y_pred_svm)
# precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
# recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
# f1_svm = f1_score(y_test, y_pred_svm, average='weighted')

# # Print results
# print("Accuracy:", round(accuracy_svm, 3))
# print("Precision:", round(precision_svm, 3))
# print("Recall:", round(recall_svm, 3))
# print("F1-Score:", round(f1_svm, 3))

# #GRADIENT BOOSTING
# g_boost = GradientBoostingClassifier(random_state=123)

# # Train the GBC on the train data
# g_boost.fit(scaled_X_train, y_train)

# # Predict on test data
# y_pred_gbc = g_boost.predict(scaled_X_test)

# # Calculate evaluation metrics
# accuracy_gbc = accuracy_score(y_test, y_pred_gbc)
# precision_gbc = precision_score(y_test, y_pred_gbc, average='weighted')
# recall_gbc = recall_score(y_test, y_pred_gbc, average='weighted')
# f1_gbc = f1_score(y_test, y_pred_gbc, average='weighted')

# # Print results
# print("Accuracy:", round(accuracy_gbc, 3))
# print("Precision:", round(precision_gbc, 3))
# print("Recall:", round(recall_gbc, 3))
# print("F1-Score:", round(f1_gbc, 3))

# #Evaluation
# models = ["Naive Bayes","Logistic Regression", "Decision Tree", "Random Forest", "KNN", "SVM", "Gradient Boosting"]
# accuracy = [accuracy_nb, accuracy_lr, accuracy_dt, accuracy_rf, accuracy_knn, accuracy_svm, accuracy_gbc]
# precision = [precision_nb, precision_lr, precision_dt, precision_rf, precision_knn, precision_svm, precision_gbc]
# recall = [recall_nb, recall_lr, recall_dt, recall_rf, recall_knn, recall_svm, recall_gbc]
# f1_score = [f1_nb, f1_lr, f1_dt, f1_rf, f1_knn, f1_svm, f1_gbc]
# import pickle
# with open('fetRisk.pickle', 'wb') as f:
#     pickle.dump(g_boost, f)

# import json
# columns = {
#     "data_columns": [col.lower() for col in df_features.columns]
# }

# with open('fetal_columns.json', 'w') as f:
#     f.write(json.dumps(columns))