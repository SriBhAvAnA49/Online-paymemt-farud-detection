import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('final_model.joblib')

# Define a function to make predictions
def predict_fraud(model, input_data):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    return prediction[0]

# Define a function for batch predictions
def batch_predict_fraud(model, df):
    predictions = model.predict(df)
    return ["Fraud" if pred == 1 else "Not Fraud" for pred in predictions]

# Set page configuration
st.set_page_config(page_title="Online Payment Fraud Detection", page_icon=":credit_card:", layout="wide")

# CSS for styling
st.markdown("""
<style>
body {
    background-color: #000000;
    color: #ffffff;
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
}
.navbar {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    padding: 10px 20px;
    background-color: #000000;
    color: #ffffff;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
.navbar-title {
    font-size: 24px;
    font-weight: bold;
}
.content {
    padding: 20px;
    background-color: #1a1a1a;
    color: #ffffff;
    margin: 20px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    border-radius: 8px;
}
.header {
    font-size: 32px;
    margin-bottom: 20px;
    color: #ffffff;
}
.footer {
    text-align: center;
    margin-top: 20px;
    color: #cccccc;
}
.stButton>button {
    background-color: #004080;
    color: #ffffff;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
    font-size: 16px;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #0059b3;
}
.tab-content {
    padding: 20px;
    border-radius: 8px;
    background-color: #333333;
    margin-top: 20px;
}
.tab-button {
    background-color: #004080;
    color: #ffffff;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
    font-size: 16px;
    font-weight: bold;
}
.tab-button:hover {
    background-color: #0059b3;
}
</style>
""", unsafe_allow_html=True)

# Navbar
st.markdown('<div class="navbar"><div class="navbar-title">Online Payment Fraud Detection</div></div>', unsafe_allow_html=True)

# Main content area
st.markdown('<div class="content">', unsafe_allow_html=True)

# Tabs
tab_home, tab_prediction, tab_batch_prediction, tab_explore = st.tabs(["🏠 Home", "⚙️ Prediction", "📊 Batch Prediction", "🔍 Explore"])

with tab_home:
    st.header("Online Payment Fraud Detection App")
    st.markdown("""
    Welcome to the Online Payment Fraud Detection App.
    
    This app predicts whether a transaction is fraudulent based on the provided details.
    
    Use the tabs above to navigate through different sections.
    """)

with tab_prediction:
    st.header("Fraud Detection")

    # Initialize session state for input fields
    if "transaction_type" not in st.session_state:
        st.session_state.transaction_type = "CASH_IN"
    if "amount" not in st.session_state:
        st.session_state.amount = 0.0
    if "oldbalanceOrg" not in st.session_state:
        st.session_state.oldbalanceOrg = 0.0
    if "newbalanceOrig" not in st.session_state:
        st.session_state.newbalanceOrig = 0.0

    # Define input fields
    transaction_type = st.radio("Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"], key="transaction_type")
    amount = st.number_input("Amount", value=st.session_state.amount, key="amount")
    oldbalanceOrg = st.number_input("Old Balance (Origin)", value=st.session_state.oldbalanceOrg, key="oldbalanceOrg")
    newbalanceOrig = st.number_input("New Balance (Origin)", value=st.session_state.newbalanceOrig, key="newbalanceOrig")
    
    # Label encode the transaction type
    transaction_type_encoded = {
        "CASH_IN": 0,
        "CASH_OUT": 1,
        "DEBIT": 2,
        "PAYMENT": 3,
        "TRANSFER": 4
    }
    
    input_data = {
        'type': transaction_type_encoded[transaction_type],
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig
    }

    # Predict button
    if st.button("Predict", key="predict_button"):
        result = predict_fraud(model, input_data)
        if result == 1:
            st.error("The transaction is predicted to be Fraudulent")
        else:
            st.success("The transaction is predicted to be Legitimate")
    
with tab_batch_prediction:
    st.header("Batch Prediction")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_uploader")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("DataFrame Loaded:")
        st.write(df.head())
        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")

        # Predict button for batch prediction
        if st.button("Predict for Batch", key="batch_predict_button"):
            predictions = batch_predict_fraud(model, df)
            df['isFraud'] = predictions

            # Show prediction counts
            fraud_count = df['isFraud'].value_counts().get("Fraud", 0)
            legit_count = df['isFraud'].value_counts().get("Not Fraud", 0)
            st.write(f"Number of Fraudulent Transactions: {fraud_count}")
            st.write(f"Number of Legitimate Transactions: {legit_count}")

            # Provide a download button for the resulting CSV
            st.write("Download the predictions as a CSV file:")
            st.download_button(
                label="Download CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name='batch_predictions.csv',
                mime='text/csv'
            )

with tab_explore:
    st.header("Overview")
    st.markdown("""
    ### Detailed Explanation
    
    This web application is designed to demonstrate online payment fraud detection using a machine learning model.
    
    #### Features:
    - **Home Tab**: Provides a brief introduction and overview of the application.
    - **Prediction Tab**: Allows users to input transaction details and predicts whether the transaction is fraudulent.
    - **Batch Prediction Tab**: Allows users to upload a CSV file, get predictions for all transactions, and download the results.
    - **Explore Tab**: You are currently on this tab, which provides more detailed explanations about each section.
    
    #### How to Use:
    - Select the transaction type (e.g., CASH_OUT, TRANSFER).
    - Enter the transaction amount, old balance (origin), and new balance (origin).
    - Click on the "Predict" button to see the prediction result.
    - Use the "Clear" button to reset the input fields.
    - For batch prediction, upload a CSV file with transaction data, click "Predict for Batch" to get predictions, and download the results.
    
    #### Additional Information:
    - This application uses a trained machine learning model (final_model.joblib) to make predictions.
    - Styling and layout are customized using Streamlit and CSS.
    
    #### About:
    Created with Streamlit.
    """)

# Close the main content area
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Created with Streamlit</div>', unsafe_allow_html=True)
