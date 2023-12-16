
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
    if 'is_training_ready' not in st.session_state:
        st.session_state['is_training_ready'] = False
    
    if 'training_model_type' not in st.session_state:
        st.session_state["training_model_type"] = "Logistic regression"

    if "training_data" not in st.session_state:
        st.session_state['training_data'] = None
    
    if "trained_model" not in st.session_state:
        st.session_state["trained_model"] = None
    
    if "testing_file_data" not in st.session_state:
        st.session_state["testing_file_data"] = None

    if "train_test_split" not in st.session_state:
        st.session_state["train_test_split"] = None, None, None, None 

    with st.sidebar:
        mode = st.selectbox(
                "Select mode",
                ("Training", "Testing")
            )

        if mode == "Training":
            training_file = st.file_uploader("Upload csv file for training", type=["csv"])
            if training_file is not None:
                st.session_state['training_data'] = pd.read_csv(training_file)

            training_model = st.selectbox(
                "Select training model",
                ("Logistic regression", "SVM", "Naive Bayes", "Decision tree", "XGBoost")
            )

            if st.button("Start Training with " + training_model) and isDataLoaded(st.session_state['training_data']):
                st.session_state["training_model_type"] = training_model
                trained_model, _ = startLearning(training_data=st.session_state['training_data'], training_model=training_model)
                st.session_state["trained_model"] = trained_model
                st.session_state['is_training_ready'] = True
        if mode == "Testing":
            upload_type = st.selectbox(
                "Select testing data upload type",
                ("File", "Inputs")
            )
    
    if mode == "Training":
        dp = DataProcessor(df=st.session_state['training_data'])
        drawer = Drawer(st.session_state['training_data'])
        if st.sidebar.button("Visualize uploaded data"):
            st.session_state['is_training_ready'] = False
            if isDataLoaded(st.session_state['training_data']) and not st.session_state['is_training_ready']:
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
        
        if st.session_state['is_training_ready']:
            predictor = Predictor()
            dp = DataProcessor(df=st.session_state['training_data'])
            X, Y = dp.prepare_data()
            X_train, X_test, Y_train, Y_test = st.session_state['train_test_split']
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

    if mode == "Testing":
        if not st.session_state['is_training_ready']:
            st.error('You need to train model before testing!', icon="🚨")
        else:
            if upload_type == "File":
                testing_file = st.file_uploader("Upload csv file for testing", type=["csv"])
                if testing_file is not None:
                    data = pd.read_csv(testing_file).iloc[:50]
                    testing_data = data.copy()
                    st.session_state["testing_file_data"] = testing_data

                    dp = DataProcessor(df=data)
                    X, _ = dp.prepare_data(False)
                    X_train, _x, Y_train, _y = st.session_state["train_test_split"]
                    x_rus, y_rus = dp.underSampling(X_train, Y_train)

                    x_test = X
                    prediction= st.session_state["trained_model"].predict(x_test)
                    data["Churn Prediction Result"] = prediction
                    st.table(data)
                    if st.session_state["training_model_type"] == "Logistic regression":
                        coefficients = st.session_state["trained_model"].coef_
                        drawer = Drawer(df=data)

                        res = drawer.plot_logreg_impact(coefficients, x_rus)
                        st.pyplot(res)
                    if st.session_state["training_model_type"] == "Decision tree":
                        importances = st.session_state["trained_model"].feature_importances_
                        drawer = Drawer(df=data)

                        res = drawer.plot_logreg_impact(importances, x_rus)

                        st.pyplot(res)
                    data["Churn"] = prediction
                    drawer = Drawer(data)
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



            if upload_type == "Inputs":
                gender = st.selectbox('Gender:', ['Мale', 'Female'])
                seniorcitizen= st.selectbox(' Customer is a senior citizen:', [0, 1])
                zipCode= st.number_input('Zip code', min_value=90020, max_value=99999, value=90020)
                partner= st.selectbox(' Customer has a partner:', ['Yes', 'No'])
                dependents = st.selectbox(' Customer has  dependents:', ['Yes', 'No'])
                tenure = st.number_input('Number of months the customer has been with the current telco provider :', min_value=0, max_value=240, value=0)
                phoneservice = st.selectbox(' Customer has phoneservice:', ['Yes', 'No'])
                multiplelines = st.selectbox(' Customer has multiplelines:', ['Yes', 'No', 'No phone service'])
                internetservice= st.selectbox(' Customer has internetservice:', ['DSL', 'No', 'Fiber optic'])
                onlinesecurity= st.selectbox(' Customer has onlinesecurity:', ['Yes', 'No', 'No internet service'])
                onlinebackup = st.selectbox(' Customer has onlinebackup:', ['Yes', 'No', 'No internet service'])
                deviceprotection = st.selectbox(' Customer has deviceprotection:', ['Yes', 'No', 'No internet service'])
                techsupport = st.selectbox(' Customer has techsupport:', ['Yes', 'No', 'No internet service'])
                streamingtv = st.selectbox(' Customer has streamingtv:', ['Yes', 'No', 'No internet service'])
                streamingmovies = st.selectbox(' Customer has streamingmovies:', ['Yes', 'No', 'No internet service'])
                contract= st.selectbox(' Customer has a contract:', ['Month-to-month', 'One year', 'Two year'])
                paperlessbilling = st.selectbox(' Customer has a paperlessbilling:', ['Yes', 'No'])
                paymentmethod= st.selectbox('Payment Option:', ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check' ,'Mailed check'])
                monthlycharges= st.number_input('Monthly charges :', min_value=0, max_value=240, value=0)
                totalcharges = tenure*monthlycharges
                churnReason= st.selectbox(' Churn reason:', ['Competitor offered higher download speeds', 'Competitor made better offer', 'Competitor had better devices', 'Competitor offered more data'])
                input_dict = {
                    'gender': [gender],
                    'SeniorCitizen': [seniorcitizen],
                    'ZIPcode': [zipCode],
                    'Partner': [partner],
                    'Dependents': [dependents],
                    'tenure': [tenure],
                    'PhoneService': [phoneservice],
                    'MultipleLines': [multiplelines],
                    'InternetService': [internetservice],
                    'OnlineSecurity': [onlinesecurity],
                    'OnlineBackup': [onlinebackup],
                    'DeviceProtection': [deviceprotection],
                    'TechSupport': [techsupport],
                    'StreamingTV': [streamingtv],
                    'StreamingMovies': [streamingmovies],
                    'Contract': [contract],
                    'PaperlessBilling': [paperlessbilling],
                    'PaymentMethod': [paymentmethod],
                    'MonthlyCharges': [monthlycharges],
                    'TotalCharges': [totalcharges],
                    'ChurnReason': [churnReason]
                }

                fullDF = pd.read_csv("./test_prediction_ds.csv")

                input_data = pd.DataFrame.from_dict(input_dict)
                input_data["customerID"] = "test-input"

                data = pd.concat([fullDF, input_data], ignore_index=True)
                data['ChurnReason'] = data['ChurnReason'].fillna('Competitor made better offer')

                dp = DataProcessor(df=data)
                X, _ = dp.prepare_data(False)
                X_train, _x, Y_train, _y = st.session_state["train_test_split"]
                x_rus, y_rus = dp.underSampling(X_train, Y_train)
                
                prediction= st.session_state["trained_model"].predict(X.dropna())
                input_data["Churn Prediction Result"] = prediction[-1]
                st.table(input_data.drop(columns="customerID"))
                if st.session_state["training_model_type"] == "Logistic regression":
                    coefficients = st.session_state["trained_model"].coef_
                    drawer = Drawer(df=data)

                    res = drawer.plot_logreg_impact(coefficients, x_rus)
                    st.pyplot(res)
                if st.session_state["training_model_type"] == "Decision tree":
                    importances = st.session_state["trained_model"].feature_importances_
                    drawer = Drawer(df=data)

                    res = drawer.plot_logreg_impact(importances, x_rus)

                    st.pyplot(res)




# def test():
#     dp = DataProcessor()
#     df = dp.fetch_file(link=LINK)
#     X, Y = dp.prepare_data()
#     X_train, X_test, Y_train, Y_test = dp.train_test_split(X, Y)
#     x_sm, y_sm = dp.overSampling(X_train, Y_train)
#     x_rus, y_rus = dp.underSampling(X_train, Y_train)
#     predictor = Predictor()
#     best_model, best_predictions = predictor.get_naive_bayes_best_model([('X_train', X_train, Y_train), ('x_rus', x_rus, y_rus), ('x_sm', x_sm, y_sm)], X_test, Y_test)
#     print(X_test.iloc[:1])
#     print(best_model.predict(X_test.iloc[:1]))

# def test1():
#     # Загрузка CSV файла в DataFrame
#     df = pd.read_csv('/Users/antonmarynych/Desktop/Учеба/dg/telecom_customer_churn_final.csv')
#     print(df)

#     # Удаление колонки с названием 'Churn'
#     df = df.drop('Churn', axis=1)  # axis=1 указывает на удаление колонки, а не строки

#     # Обрезка первых пяти строк
#     df = df.iloc[5:]

#     # Сохранение измененного DataFrame обратно в CSV файл
#     df.iloc[:5].to_csv('test_prediction_ds_1.csv', index=False)  # index=False для сохранения


if __name__ == '__main__':
	main()