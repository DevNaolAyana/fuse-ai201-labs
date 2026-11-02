import os
import requests
import streamlit as st

# Point to your local FastAPI. (In cloud deployment, this would be your Render URL)
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Fraud Risk Demo", layout="wide")

st.title("💳 Credit Card Fraud Analyzer")
st.caption("Educational demonstration. Not for real banking decisions.")

with st.expander("About this tool: intended use and limitations"):
    st.markdown(
        "**Intended users:** Fraud analysts and reviewers learning how the model behaves.\n\n"
        "**What the result means:** A model's estimate, based on historical data, "
        "of the probability that a transaction is fraudulent.\n\n"
        "**Limitations:** The model was trained on a limited European dataset from 2013. "
        "It does not replace professional judgment, and it must not be the only "
        "basis for freezing a real customer's account."
    )

with st.sidebar:
    st.header("Transaction Information")
    amount = st.number_input("Transaction Amount ($)", 0.0, 50000.0, 239.93, 10.0, help="The monetary value of the transaction.")
    time = st.number_input("Time (Seconds)", 0.0, 200000.0, 406.0, 100.0, help="Seconds elapsed since the first transaction in the dataset.")
    
    st.write("---")
    st.write("PCA Components (V1-V28)")
    st.write("For this demo, we will leave these at a default safe value.")
    
    submit = st.button("Analyze Transaction", type="primary", use_container_width=True)

if submit:
    # We create a payload with the features.
    # For the 28 PCA features, we will just pass 0.0 to keep the demo simple.
    payload = {
        "Time": time,
        "Amount": amount,
        **{f"V{i}": 0.0 for i in range(1, 29)}
    }

    try:
        with st.spinner("Contacting the API... (A sleeping service may take up to a minute)"):
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
            
    except requests.exceptions.Timeout:
        st.error("The service took too long to answer. Please wait a moment and try again.")
        st.stop()
    except requests.exceptions.RequestException:
        st.error("Could not reach the service. Is your FastAPI server running?")
        st.stop()

    if r.status_code == 200:
        try:
            result = r.json()
            prob = float(result["fraud_probability"])
            decision = str(result["decision"])
        except (ValueError, KeyError, TypeError):
            st.error("The service sent an answer that could not be read.")
            st.stop()

        c1, c2 = st.columns(2)
        c1.metric("Estimated Fraud Probability", f"{prob:.1%}")
        
        # Color-code the decision text
        if decision == "Fraud":
            c2.error(f"Decision: {decision.upper()}")
        else:
            c2.success(f"Decision: {decision.upper()}")

        st.write(
            f"The model estimates about a **{prob:.0%}** chance that this transaction is fraudulent. "
            f"The suggested decision is **{decision}**."
        )
        st.info("This is a model estimate to support, not replace, human judgment.")

    elif r.status_code == 422:
        st.warning("Some inputs are invalid. Please check the values.")
    else:
        st.error("The service returned an error. Please try again later.")
else:
    st.write("Enter transaction details in the sidebar and press **Analyze Transaction**.")