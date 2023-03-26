import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# Define the Streamlit app
def app():
    # Set the app title
    st.title("Transaction Fraud Detector")

    # Allow the user to upload a CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    # If a file was uploaded, load the data and display it
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)

        # Split the data into features and labels
        X = data[['TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS']]
        y = data['TX_FRAUD']

        # Train the decision tree classifier
        clf = DecisionTreeClassifier()
        clf.fit(X, y)

        # Allow the user to enter transaction data
        st.write("Enter transaction data:")
        TX_AMOUNT = st.number_input("Transaction Amount")
        TX_TIME_SECONDS = st.number_input("Transaction Time (Seconds)", step=1)
        TX_TIME_DAYS = st.number_input("Transaction Time (Days)", step=1)

        # Add a button to make the prediction
        if st.button("Make Prediction"):
            # Make a prediction for the new transaction
            new_transaction = [[TX_AMOUNT, TX_TIME_SECONDS, TX_TIME_DAYS]]
            prediction = clf.predict(new_transaction)

            # Display the prediction
            if prediction[0] == 0:
                st.write("The transaction is not fraudulent.")
            else:
                st.write("The transaction is fraudulent.")


if __name__ == '__main__':
    app()