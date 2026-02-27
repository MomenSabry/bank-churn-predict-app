import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦", layout="wide")

st.title("🏦 Bank Customer Churn Predictor")

st.markdown("**Predict whether a customer will leave the bank — powered by your Random Forest model**")

# ====================== LOAD MODEL ======================
try:
    model = joblib.load("app.pki")
    st.success("✅ Model loaded successfully! (RandomForestClassifier)")
except FileNotFoundError:
    st.error("❌ Model file 'app.pki' not found. Place it in the same folder.")
    st.stop()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ====================== TABS ======================
tab1, tab2 = st.tabs(["🔮 Single Prediction", "📊 Batch Prediction"])

with tab1:
    st.subheader("Customer Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        credit_score = st.number_input("Credit Score", min_value=350, max_value=850, value=650)
        age = st.number_input("Age", min_value=18, max_value=92, value=38)
        tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=5)
        num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
    
    with col2:
        balance = st.number_input("Balance (€)", min_value=0.0, max_value=300000.0, value=0.0, step=1000.0)
        estimated_salary = st.number_input("Annual Salary (€)", min_value=0.0, max_value=200000.0, value=100000.0, step=1000.0)
        has_cr_card = st.selectbox("Has Credit Card?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        is_active = st.selectbox("Is Active Member?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    with col3:
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])

    if st.button("🚀 Predict Churn", type="primary", use_container_width=True):
        try:
            # Convert to model format
            geo_germany = 1 if geography == "Germany" else 0
            geo_spain = 1 if geography == "Spain" else 0
            gen_male = 1 if gender == "Male" else 0

            input_df = pd.DataFrame([[
                credit_score, age, tenure, balance, num_products,
                has_cr_card, is_active, estimated_salary,
                geo_germany, geo_spain, gen_male
            ]], columns=[
                'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                'Geography_Germany', 'Geography_Spain', 'Gender_Male'
            ])

            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1] * 100   
            
            if prediction == 1:
                st.error(f"⚠️ **HIGH RISK — Customer will likely CHURN**  \nConfidence: **{probability:.1f}%**")
            else:
                st.success(f"✅ **LOW RISK — Customer will likely STAY**  \nConfidence: **{probability:.1f}%**")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.info("Check that all inputs are within valid ranges.")

with tab2:
    st.subheader("Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV (can contain raw Geography & Gender columns)", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("**Preview of uploaded data:**")
            st.dataframe(df.head())

            # Auto process if raw categorical columns exist
            if 'Geography' in df.columns and 'Gender' in df.columns:
                df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

            required_cols = [
                'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                'Geography_Germany', 'Geography_Spain', 'Gender_Male'
            ]

            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                st.error(f"❌ Missing columns: {missing}")
                st.info("Make sure your CSV has the required features (or raw Geography + Gender).")
            else:
                X = df[required_cols]
                preds = model.predict(X)
                probs = model.predict_proba(X)[:, 1] * 100

                df['Churn_Prediction'] = preds
                df['Churn_Probability_%'] = probs.round(1)
                df['Final_Prediction'] = df['Churn_Prediction'].map({0: 'Will Stay', 1: 'Will Churn'})

                st.success("✅ Batch prediction completed!")
                st.dataframe(df)

                csv_out = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Results",
                    data=csv_out,
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"❌ Error processing CSV: {e}")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("Model Information")
    st.write("**Algorithm:** Random Forest Classifier")
    st.write("**Trained with:** SMOTE (to handle class imbalance)")
    st.write("**Dataset:** Bank Churn Modelling")
    st.caption("Built Moamen Sabry")

