import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split

st.title("Find out if you have PCOS!")

st.image("pcos.jpg",width=400)

st.markdown("Enter your personal data.")

pcos = pd.read_csv('CLEAN- PCOS SURVEY SPREADSHEET.csv')
df = pd.DataFrame(pcos)
df.rename(columns={'Age (in Years)': 'Age'}, inplace=True)
df.rename(columns={'Weight (in Kg)': 'Weight'}, inplace=True)
df.rename(columns={'Height (in Cm / Feet)': 'Height'}, inplace=True)
df.rename(columns={'After how many months do you get your periods?\n(select 1- if every month/regular)': 'period_frequency'}, inplace=True)
df.rename(columns={'Can you tell us your blood group ?': 'blood_group'}, inplace=True)
df.rename(columns={'Have you gained weight recently?': 'weight_increased'}, inplace=True)
df.rename(columns={'Do you have excessive body/facial hair growth ?': 'excessive_hair'}, inplace=True)
df.rename(columns={'Are you noticing skin darkening recently?': 'skin_darkening'}, inplace=True)
df.rename(columns={'Do have hair loss/hair thinning/baldness ?': 'baldness'}, inplace=True)
df.rename(columns={'Do you eat fast food regularly ?': 'fast_food'}, inplace=True)
df.rename(columns={'Do you exercise on a regular basis ?': 'exercise'}, inplace=True)
df.rename(columns={'Do you experience mood swings ?': 'mood_swings'}, inplace=True)
df.rename(columns={'Are your periods regular ?': 'regular_period'}, inplace=True)
df.rename(columns={'How long does your period last ? (in Days)\nexample- 1,2,3,4.....': 'period_days'}, inplace=True)
df.rename(columns={'Have you been diagnosed with PCOS/PCOD?': 'PCOS'}, inplace=True)
df.rename(columns={'Do you have pimples/acne on your face/jawline ?': 'pimples'}, inplace=True)


df = df.drop('period_frequency',axis=1)

variables = df.drop('PCOS', axis=1)
target = df['PCOS']


X_train, X_val, y_train, y_val = train_test_split(variables, 
                                                    target, 
                                                    train_size = 0.7, 
                                                    random_state = 0, 
                                                    stratify = target, 
                                                    shuffle = True)




#load best model
def load_model():
        model = DecisionTreeClassifier(max_leaf_nodes=13,max_depth=5).fit(X_train,y_train)
        return model

model = load_model()

# Function to make predictions
def make_prediction(input_data):
    # Make prediction
    prediction = model.predict(input_data)
    return prediction
# Questions for user input
age = st.number_input('Enter your age',step=1, format='%d')
weight = st.number_input('Enter your weight(in kg)',step=0.1, format='%0.1f')
height = st.number_input('Enter your height(in cm)',step=1 format='%d')
period_days = st.number_input('How long does your period last ? (in Days)\nexample- 1,2,3,4.....',step=1,format='%d')
st.image("blood_group.jpg",width=400)
blood_group = st.selectbox('Enter your blood group: 11 indicates A+, 12 indicates A-, 13 indicates B+, 14 indicates B-, 15 indicates O+, 16 indicates O-, 17 indicates AB+, 18 indicates AB-',['11','12','13','14','15','16','17','18'])
   
st.image("gain.jpg",width=400)
   
weight_increased = st.selectbox('Have you gained weight recently? (select "0" for no and "1" for yes)',['0','1'])

st.image("excesshair.png",width=400)

excessive_hair = st.selectbox('Do you have excessive body/facial hair growth? (select "0" for no and "1" for yes)',['0','1'])

st.image("skindarkening.jpg",width=350)

skin_darkening = st.selectbox('Are you noticing any skin darkening recently? (select "0" for no and "1" for yes)',['0','1'])

st.image("hair.jpg",width=400)

baldness = st.selectbox('Do you have hair loss/baldness or hair thining? (select "0" for no and "1" for yes)',['0','1'])

st.image("fastfood.jpg",width=400)

fast_food = st.selectbox('Do you eat fast food regularly? (select "0" for no and "1" for yes)',['0','1'])

st.image("moodswings.jpg",width=400)

mood_swings = st.selectbox('Do you experience mood swings? (select "0" for no and "1" for yes)',['0','1'])

st.image("exercise.jpeg",width=400)

exercise = st.selectbox('Do you exercise on a regular basis? (select "0" for no and "1" for yes)',['0','1'])

st.image("period.jpg",width=400)

regular_period = st.selectbox('Are your periods regular? (select "0" for no and "1" for yes)',['0','1'])

st.image("pimples.jpg",width=400)

pimples = st.selectbox('Do you have pimples/acne on your chin/jawline? (select "0" for no and "1" for yes)',['0','1'])

st.markdown("Click 'predict' to see your results.")

    # Button to trigger prediction
if st.button('Predict'):
    # Create input data as a DataFrame
    input_data = pd.DataFrame({'Age': [age], 'Weight': [weight], 'Height': [height], 'blood_group': [blood_group], 'weight_increased': [weight_increased], 'excessive_hair': [excessive_hair], 'skin_darkening': [skin_darkening], 'baldness': [baldness], 'pimples': [pimples], 'fast_food': [fast_food], 'exercise': [exercise], 'mood_swings': [mood_swings], 'regular_period': [regular_period],'period_days': [period_days] })
    # Make prediction
    prediction = make_prediction(input_data)
    # Display prediction
    st.write('Prediction:', prediction)
    if prediction == 0:
        st.write("Our system predicted that you are not ailing from PCOS.\n Although our test does not intend to replace doctor's tests so if you have second thoughts you are highly encouraged to visit your gynecologist!")
    else:
         st.write("Our system's prediction is that you are ailing from PCOS. You are higly recommended to visit your gynecologist for the conferming test and treatment")
