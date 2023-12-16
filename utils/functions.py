from utils.data_processor import DataProcessor
from utils.predictor import Predictor
import streamlit as st

def isDataLoaded(df):
    return df is not None and not df.empty

def startLearning(training_model, training_data):
    predictor = Predictor()
    dp = DataProcessor(df=training_data)
    X, Y = dp.prepare_data()
    X_train, X_test, Y_train, Y_test = dp.train_test_split(X, Y)
    st.session_state["train_test_split"] = X_train, X_test, Y_train, Y_test
    x_sm, y_sm = dp.overSampling(X_train, Y_train)
    x_rus, y_rus = dp.underSampling(X_train, Y_train)
    if training_model == "Logistic regression":
        return predictor.get_logreg_best_model(
            [
                ('X_train', X_train, Y_train), 
                ('x_rus', x_rus, y_rus), 
                ('x_sm', x_sm, y_sm)
            ], 
            X_test, 
            Y_test
        )
    if training_model == "SVM":
        return predictor.get_svm_best_model(
            [
                ('X_train', X_train, Y_train), 
                ('x_rus', x_rus, y_rus), 
                ('x_sm', x_sm, y_sm)
            ], 
            X_test, 
            Y_test
        )
    if training_model == "Naive Bayes":
        return predictor.get_naive_bayes_best_model(
            [
                ('X_train', X_train, Y_train), 
                ('x_rus', x_rus, y_rus), 
                ('x_sm', x_sm, y_sm)
            ], 
            X_test, 
            Y_test
        )
    if training_model == "Decision tree":
        return predictor.get_decision_tree_best_model(
            [
                ('X_train', X_train, Y_train), 
                ('x_rus', x_rus, y_rus), 
                ('x_sm', x_sm, y_sm)
            ], 
            X_test, 
            Y_test
        )
    if training_model == "XGBoost":
        return predictor.get_xgbclassifier_best_model(
            [
                ('X_train', X_train, Y_train), 
                ('x_rus', x_rus, y_rus), 
                ('x_sm', x_sm, y_sm)
            ], 
            X_test, 
            Y_test
        )
    return 