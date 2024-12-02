# disease-prediction-model
3 MLM to prdicted the probability that a patient has the following: asthma, heart disease and diabetese. The models rely on laboratory data as input to give the predictions
the models are trained in main.py and asthma_model.py using datasets for each disease. the datasets have information about patients that have the disease and ones that do not, blood works, blood sugar levels, among other data needed to predict possibility of having the disease. the models are trained using randomforrest classifier and the models are saved. Main.py trains diabetes model and heart disease model and saves the models as diabetes_model.pkl and heart_disease_model.plk
the asthma_model.py trains asthma_multi_target_model.pkl which predicts three possible outcomes severity moderate, severity mild and severity none
app.py is the front end which uses streamlit to localy host a webpage
