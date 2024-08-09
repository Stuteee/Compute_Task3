import pickle
import streamlit as st
import pandas as pd

# Load your trained model using joblib
try:
    model = pickle.load('xgb_model.sav')
    st.write("Model loaded successfully!")
except Exception as e:
    st.write(f"Error loading model: {e}")
    st.stop()

# Define risk category mapping
income_mapping = {
    1: '>50K',
    0: '<=50K'
}

# Define the Streamlit app
def main():
    st.title('Income Prediction')

    # Create inputs for numerical and float columns
    age = st.number_input('Age', min_value=0, value=100)
    education_num = st.number_input('Years of Education', min_value=0, value=20)
    capital_gain = st.number_input('Gain in Capital', min_value=0, value=10000)
    capital_loss = st.number_input('Loss in Capital', min_value=0, value=10000)  # Changed to selectbox
    hours_per_week = st.number_input('Hours per Week', min_value=0, value=100)

    # One-hot encoding for the 'workclass' column
    workclasss = ['Private', 'Government', 'Self_Employed', 'Not_Employed']
    workclass = st.selectbox('Workclass', workclasss)

    # Initialize one-hot encoded features with zeros
    one_hot_encoded_purposes = [0] * len(workclasss)
    workclass_index = workclasss.index(workclass)
    one_hot_encoded_purposes[workclass_index] = 1

    # One-hot encoding for the 'Purpose' column
    maritals  = ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed','Married-spouse-absent','Married-AF-spouse']
    marital = st.selectbox('Marital_Status', maritals)

    # Initialize one-hot encoded features with zeros
    one_hot_encoded_maritals = [0] * len(maritals)
    marital_index = maritals.index(marital)
    one_hot_encoded_maritals[marital_index] = 1

    # One-hot encoding for the 'Purpose' column
    relationships  = ['Not-in-family', 'Unmarried', 'Own-child', 'Other-relative', 'Husband', 'Wife']
    relationship = st.selectbox('Relationship', relationships)

    # Initialize one-hot encoded features with zeros
    one_hot_encoded_relationships = [0] * len(relationships)
    relationship_index = relationships.index(relationship)
    one_hot_encoded_relationships[relationship_index] = 1

    # Create a DataFrame with user inputs and include all columns
    data = pd.DataFrame({
        'Age': [age],
        'Years of Education': [education_num],
        'Gain in Capital': [capital_gain],
        'Loss in Capital': [capital_loss],
        'Hours per Week': [hours_per_week],
        'Workclass_Private': [one_hot_encoded_purposes[0]],
        'Workclass_Government': [one_hot_encoded_purposes[1]],
        'Workclass_Self_Employed': [one_hot_encoded_purposes[2]],
        'Workclass_Not_Employed': [one_hot_encoded_purposes[3]],
        'Marital_Status_Married-civ-spouse': [one_hot_encoded_maritals[0]],
        'Marital_Status_Never-married': [one_hot_encoded_maritals[1]],
        'Marital_Status_Divorced': [one_hot_encoded_maritals[2]],
        'Marital_Status_Separated': [one_hot_encoded_maritals[3]],
        'Marital_Status_Widowed': [one_hot_encoded_maritals[4]],
        'Marital_Status_Married-spouse-absent': [one_hot_encoded_maritals[5]],
        'Marital_Status_Married-AF-spouse': [one_hot_encoded_maritals[6]],
        'Relationship_Not-in-family': [one_hot_encoded_relationships[0]],
        'Relationship_Unmarried': [one_hot_encoded_relationships[1]],
        'Relationship_Own-child': [one_hot_encoded_relationships[2]],
        'Relationship_Other-relative': [one_hot_encoded_relationships[3]],
        'Relationship_Husband': [one_hot_encoded_relationships[4]],
        'Relationship_Wife': [one_hot_encoded_relationships[5]]
    })

    # Debug: Display DataFrame
    st.write("Input Data for Prediction:")
    st.write(data)

    # Create a button to trigger prediction
    if st.button('Predict'):
        try:
            # Make prediction
            prediction = model.predict(data)
            income_category = income_mapping.get(prediction[0], 'Unknown Income Category')
            st.write(f'Predicted Income Category: {income_category}')
        except Exception as e:
            st.write(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()