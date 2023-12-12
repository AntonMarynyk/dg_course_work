
import streamlit as st
import pandas as pd
from utils.data_processor import DataProcessor
from utils.drawer import Drawer
from utils.predictor import Predictor
from utils.functions import isDataLoaded, startLearning

# Link to initial file with ds
LINK = 'https://drive.google.com/uc?id=1tbgtf2us4bmZXWUDk9_EpgevwA1sI3Ri&export=download'
DEM_COL = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
ACC_COL = ['Contract', 'PaperlessBilling', 'PaymentMethod']
SERV_COL = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    training_data = None
    training_model = ""
    upload_type=""
    is_prediction_result_ready = False

    with st.sidebar:
        mode = st.selectbox(
                "Select mode",
                ("Training", "Testing")
            )

        if mode == "Training":
            training_file = st.file_uploader("Upload csv file for training", type=["csv"])
            if training_file is not None:
                training_data = pd.read_csv(training_file)

            training_model = st.selectbox(
                "Select training model",
                ("Logistic regression", "SVM", "Naive Bayes", "Decision tree", "XGBoost")
            )

            if st.button("Start Training with " + training_model) and isDataLoaded(training_data):
                startLearning(training_data=training_data, training_model=training_model)
                is_prediction_result_ready=True
        if mode == "Testing":
            upload_type = st.selectbox(
                "Select testing data upload type",
                ("File", "Inputs")
            )
    
    if mode == "Training":
        dp = DataProcessor(df=training_data)
        drawer = Drawer(training_data)
        if st.sidebar.button("Visualize uploaded data") and isDataLoaded(training_data) and not is_prediction_result_ready:
            is_prediction_result_ready=False
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Churn distribution", 
                "Gender Churn distribution", 
                "Churn Reason distribution",
                "Column Histogram",
                "Box plot",
                "Smooth",
            ])

            with tab1:
                churn_dist = drawer.plot_churn_dist()
                st.pyplot(churn_dist)
            with tab2:
                gender_churn_dist = drawer.plot_gender_churn_dist()
                st.plotly_chart(gender_churn_dist)
            with tab3:
                churn_reason_dist = drawer.plot_churn_reason_dist()
                st.pyplot(churn_reason_dist)
            with tab4:
                columns_histogram = drawer.plot_columns_histogram(SERV_COL)
                st.pyplot(columns_histogram)
            with tab5:
                boxplots = drawer.plot_boxplots()
                st.pyplot(boxplots)
            with tab6:
                smooth_dist = drawer.plot_smooth_dist()
                st.pyplot(smooth_dist)
        
        if is_prediction_result_ready:
            predictor = Predictor()
            dp = DataProcessor(df=training_data)
            X, Y = dp.prepare_data()
            X_train, X_test, Y_train, Y_test = dp.train_test_split(X, Y)
            plt, table = predictor.plot_prediction_result(X_test.shape[0])
            st.pyplot(plt)
            st.write("Пояснення метрик якості моделі:")
            st.write("-------------------------------------------------")
            st.write("Accuracy (Точність): Відсоток правильних класифікацій.")
            st.write("F1 Score: Гармонічний середній між точністю та повнотою.")
            st.write("Precision (Точність): Відсоток правильно класифікованих позитивних прикладів.")
            st.write("Recall (Повнота): Відсоток правильно визначених позитивних прикладів.")
            st.write("-------------------------------------------------\n")
            st.table(table)



    if mode == "Testing" and upload_type == "File":
        testing_file = st.file_uploader("Upload csv file for testing", type=["csv"])
        if testing_file is not None:
            data = pd.read_csv(testing_file)

    if mode == "Testing" and upload_type == "Inputs":
        gender = st.selectbox('Gender:', ['male', 'female'])
        seniorcitizen= st.selectbox(' Customer is a senior citizen:', [0, 1])
        partner= st.selectbox(' Customer has a partner:', ['yes', 'no'])
        dependents = st.selectbox(' Customer has  dependents:', ['yes', 'no'])
        phoneservice = st.selectbox(' Customer has phoneservice:', ['yes', 'no'])
        multiplelines = st.selectbox(' Customer has multiplelines:', ['yes', 'no', 'no_phone_service'])
        internetservice= st.selectbox(' Customer has internetservice:', ['dsl', 'no', 'fiber_optic'])
        onlinesecurity= st.selectbox(' Customer has onlinesecurity:', ['yes', 'no', 'no_internet_service'])
        onlinebackup = st.selectbox(' Customer has onlinebackup:', ['yes', 'no', 'no_internet_service'])
        deviceprotection = st.selectbox(' Customer has deviceprotection:', ['yes', 'no', 'no_internet_service'])
        techsupport = st.selectbox(' Customer has techsupport:', ['yes', 'no', 'no_internet_service'])
        streamingtv = st.selectbox(' Customer has streamingtv:', ['yes', 'no', 'no_internet_service'])
        streamingmovies = st.selectbox(' Customer has streamingmovies:', ['yes', 'no', 'no_internet_service'])
        contract= st.selectbox(' Customer has a contract:', ['month-to-month', 'one_year', 'two_year'])
        paperlessbilling = st.selectbox(' Customer has a paperlessbilling:', ['yes', 'no'])
        paymentmethod= st.selectbox('Payment Option:', ['bank_transfer_(automatic)', 'credit_card_(automatic)', 'electronic_check' ,'mailed_check'])
        tenure = st.number_input('Number of months the customer has been with the current telco provider :', min_value=0, max_value=240, value=0)
        monthlycharges= st.number_input('Monthly charges :', min_value=0, max_value=240, value=0)
        totalcharges = tenure*monthlycharges
        output= ""
        output_prob = ""
        input_dict={
                "gender":gender ,
                "seniorcitizen": seniorcitizen,
                "partner": partner,
                "dependents": dependents,
                "phoneservice": phoneservice,
                "multiplelines": multiplelines,
                "internetservice": internetservice,
                "onlinesecurity": onlinesecurity,
                "onlinebackup": onlinebackup,
                "deviceprotection": deviceprotection,
                "techsupport": techsupport,
                "streamingtv": streamingtv,
                "streamingmovies": streamingmovies,
                "contract": contract,
                "paperlessbilling": paperlessbilling,
                "paymentmethod": paymentmethod,
                "tenure": tenure,
                "monthlycharges": monthlycharges,
                "totalcharges": totalcharges
            }


def test():
    dp = DataProcessor()
    df = dp.fetch_file(link=LINK)
    X, Y = dp.prepare_data()
    X_train, X_test, Y_train, Y_test = dp.train_test_split(X, Y)
    x_sm, y_sm = dp.overSampling(X_train, Y_train)
    x_rus, y_rus = dp.underSampling(X_train, Y_train)
    predictor = Predictor()
    best_model, best_predictions = predictor.get_xgbclassifier_best_model([('X_train', X_train, Y_train), ('x_rus', x_rus, y_rus), ('x_sm', x_sm, y_sm)], X_test, Y_test)
    print(best_model)

	

if __name__ == '__main__':
	main()