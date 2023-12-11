from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import warnings

class Predictor:
    def get_logreg_best_model(self, datasets, X_test, y_test):
        best_accuracy = 0
        best_model = None
        best_predictions = None


        for dataset_name, X_train, y_train in datasets:

            # Define hyperparameter grid
            param_grid = {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }

            # Train and predict with default parameters
            model = LogisticRegression(max_iter=3000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f'Default parameters - {dataset_name} Accuracy: {accuracy}')

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_predictions = y_pred

            # Train and predict with hyperparameter tuning
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                grid_search = GridSearchCV(LogisticRegression(max_iter=3000), param_grid, cv=3)
                grid_search.fit(X_train, y_train)

            best_model_grid = grid_search.best_estimator_
            y_pred_grid = best_model_grid.predict(X_test)
            accuracy_grid = accuracy_score(y_test, y_pred_grid)
            print(f'Best hyperparameters - {dataset_name} Accuracy: {accuracy_grid}')
            print(f'Best hyperparameters for {dataset_name}: {grid_search.best_params_}')

            if accuracy_grid > best_accuracy:
                best_accuracy = accuracy_grid
                best_model = best_model_grid
                best_predictions = y_pred_grid

        return best_model, best_predictions

    def get_svm_best_model(self, datasets, X_test, y_test, hyperparameter_tuning=True):
        best_accuracy = 0
        best_model = None
        best_predictions = None

        for dataset_name, X_train, y_train in datasets:

            # Define hyperparameter grid
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': [0.1, 0.01, 0.001],
                'kernel': ['linear', 'rbf', 'poly']
            }

            # Train and predict with default parameters
            model = SVC()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f'Default parameters - {dataset_name} Accuracy: {accuracy}')

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_predictions = y_pred

            # Train and predict with hyperparameter tuning
            if hyperparameter_tuning:
                grid_search = GridSearchCV(SVC(), param_grid, cv=3)
                grid_search.fit(X_train, y_train)

                best_model_grid = grid_search.best_estimator_
                y_pred_grid = best_model_grid.predict(X_test)
                accuracy_grid = accuracy_score(y_test, y_pred_grid)
                print(f'Best hyperparameters - {dataset_name} Accuracy: {accuracy_grid}')
                print(f'Best hyperparameters for {dataset_name}: {grid_search.best_params_}')

                if accuracy_grid > best_accuracy:
                    best_accuracy = accuracy_grid
                    best_model = best_model_grid
                    best_predictions = y_pred_grid

        return best_model, best_predictions
    
    def get_naive_bayes_best_model(self, datasets, X_test, y_test, hyperparameter_tuning=True):
        best_accuracy = 0
        best_model = None
        best_predictions = None

        for dataset_name, X_train, y_train in datasets:
            # Naive Bayes does not require feature scaling, so we skip it

            # Train and predict with default parameters
            model = GaussianNB()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f'Default parameters - {dataset_name} Accuracy: {accuracy}')

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_predictions = y_pred

        return best_model, best_predictions

    def get_decision_tree_best_model(self, datasets, X_test, y_test, hyperparameter_tuning=True):
        best_accuracy = 0
        best_model = None
        best_predictions = None



        for dataset_name, X_train, y_train in datasets:

            # Define hyperparameter grid
            param_grid = {
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy'],
                'max_features': [None, 'sqrt', 'log2']
            }

            # Train and predict with default parameters
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f'Default parameters - {dataset_name} Accuracy: {accuracy}')

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_predictions = y_pred

            # Train and predict with hyperparameter tuning
            if hyperparameter_tuning:
                grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=3)
                grid_search.fit(X_train, y_train)

                best_model_grid = grid_search.best_estimator_
                y_pred_grid = best_model_grid.predict(X_test)
                accuracy_grid = accuracy_score(y_test, y_pred_grid)
                print(f'Best hyperparameters - {dataset_name} Accuracy: {accuracy_grid}')
                print(f'Best hyperparameters for {dataset_name}: {grid_search.best_params_}')

                if accuracy_grid > best_accuracy:
                    best_accuracy = accuracy_grid
                    best_model = best_model_grid
                    best_predictions = y_pred_grid

        return best_model, best_predictions

    def get_xgbclassifier_best_model(self, datasets, X_test, y_test, hyperparameter_tuning=True):
        best_accuracy = 0
        best_model = None
        best_predictions = None


        for dataset_name, X_train, y_train in datasets:


            # Define hyperparameter grid
            param_grid = {
                'max_depth': [3, 6, 9],
                'learning_rate': [0.1, 0.01, 0.001],
                'n_estimators': [100, 200, 300],
                'subsample': [0.8, 0.9, 1.0]
            }

            # Train and predict with default parameters
            model = XGBClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f'Default parameters - {dataset_name} Accuracy: {accuracy}')

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_predictions = y_pred

            # Train and predict with hyperparameter tuning
            if hyperparameter_tuning:
                grid_search = GridSearchCV(XGBClassifier(), param_grid, cv=3)
                grid_search.fit(X_train, y_train)

                best_model_grid = grid_search.best_estimator_
                y_pred_grid = best_model_grid.predict(X_test)
                accuracy_grid = accuracy_score(y_test, y_pred_grid)
                print(f'Best hyperparameters - {dataset_name} Accuracy: {accuracy_grid}')
                print(f'Best hyperparameters for {dataset_name}: {grid_search.best_params_}')

                if accuracy_grid > best_accuracy:
                    best_accuracy = accuracy_grid
                    best_model = best_model_grid
                    best_predictions = y_pred_grid

        return best_model, best_predictions

    