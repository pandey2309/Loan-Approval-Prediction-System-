import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ==========================
# Page Config
# ==========================

st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide"
)

# ==========================
# Load Dataset & Model
# ==========================

df = pd.read_csv("dataset/loan_approval_dataset.csv")
model = joblib.load("models/loan_model.pkl")

# Feature Engineering
df["TotalIncome"] = (
    df["ApplicantIncome"] +
    df["CoapplicantIncome"]
)

# ==========================
# Sidebar
# ==========================

st.sidebar.title("🏦 Loan Approval System")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dataset Overview",
        "📈 Data Visualizations",
        "🤖 Model Performance",
        "📝 Loan Prediction",
        "ℹ️ About Project"
    ]
)

# ==========================
# HOME PAGE
# ==========================

if page == "🏠 Home":

    st.title("🏦 ML-Based Loan Approval Prediction System")

    st.markdown("""
    This project predicts whether a loan application is likely to be approved
    based on applicant information using Machine Learning.
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", len(df))
    col2.metric("Features", 12)
    col3.metric("Final Model", "Random Forest")
    col4.metric("Accuracy", "99.81%")

    st.markdown("---")

    st.subheader("Project Highlights")

    st.write("""
    ✅ Machine Learning Based Prediction

    ✅ Random Forest Classifier

    ✅ Interactive Dashboard

    ✅ Real-Time Loan Eligibility Check

    ✅ Data Visualization and Analysis
    """)

# ==========================
# DATASET OVERVIEW
# ==========================

elif page == "📊 Dataset Overview":

    st.title("📊 Dataset Overview")

    st.subheader("Dataset Shape")

    st.write(df.shape)

    st.subheader("First 10 Records")

    st.dataframe(df.head(10))

    st.subheader("Statistical Summary")

    st.dataframe(df.describe())

# ==========================
# VISUALIZATIONS
# ==========================

elif page == "📈 Data Visualizations":

    st.title("📈 Data Visualizations")

    # Loan Status Distribution

    st.subheader("Loan Approval Distribution")

    loan_counts = df["Loan_Status"].value_counts()

    fig, ax = plt.subplots()

    ax.pie(
        loan_counts,
        labels=loan_counts.index,
        autopct="%1.1f%%"
    )

    st.pyplot(fig)

    # Property Area Distribution

    st.subheader("Property Area Distribution")

    fig2, ax2 = plt.subplots()

    df["Property_Area"].value_counts().plot(
        kind="bar",
        ax=ax2
    )

    st.pyplot(fig2)

    # Education Distribution

    st.subheader("Education Distribution")

    fig3, ax3 = plt.subplots()

    df["Education"].value_counts().plot(
        kind="bar",
        ax=ax3
    )

    st.pyplot(fig3)

# ==========================
# MODEL PERFORMANCE
# ==========================

elif page == "🤖 Model Performance":

    st.title("🤖 Model Performance")

    performance = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest"
            ],
            "Accuracy (%)": [
                98.38,
                99.94,
                99.81
            ]
        }
    )

    st.dataframe(performance)

    st.success(
        "Final Selected Model: Random Forest"
    )

    st.write("""
    Random Forest was selected because it provides
    excellent accuracy while reducing overfitting
    compared to a single Decision Tree.
    """)

# ==========================
# LOAN PREDICTION
# ==========================

elif page == "📝 Loan Prediction":

    st.title("📝 Loan Prediction")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        married = st.selectbox(
            "Married",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            [0, 1, 2, 3]
        )

        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )

        self_employed = st.selectbox(
            "Self Employed",
            ["Yes", "No"]
        )

    with col2:

        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0
        )

        coapplicant_income = st.number_input(
            "Coapplicant Income",
            min_value=0
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=1
        )

        loan_term = st.selectbox(
            "Loan Amount Term",
            [120, 180, 240, 300, 360]
        )

        credit_history = st.selectbox(
            "Credit History",
            [0, 1]
        )

        property_area = st.selectbox(
            "Property Area",
            ["Rural", "Semiurban", "Urban"]
        )

    if st.button("Predict Loan Status"):

        gender = 1 if gender == "Male" else 0
        married = 1 if married == "Yes" else 0
        education = 0 if education == "Graduate" else 1
        self_employed = 1 if self_employed == "Yes" else 0

        property_map = {
            "Rural": 0,
            "Semiurban": 1,
            "Urban": 2
        }

        property_area = property_map[property_area]

        total_income = (
            applicant_income +
            coapplicant_income
        )

        input_data = pd.DataFrame([[
            gender,
            married,
            dependents,
            education,
            self_employed,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_term,
            credit_history,
            property_area,
            total_income
        ]])

        prediction = model.predict(input_data)[0]

        confidence = model.predict_proba(
            input_data
        )[0]

        if prediction == 1:

            st.success(
                "✅ Loan Approved"
            )

            st.info(
                f"Confidence Score: {max(confidence)*100:.2f}%"
            )

        else:

            st.error(
                "❌ Loan Rejected"
            )

            st.info(
                f"Confidence Score: {max(confidence)*100:.2f}%"
            )

# ==========================
# ABOUT PAGE
# ==========================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About Project")

    st.write("""
    ML-Based Loan Approval Prediction System
    is a machine learning project developed
    using Python, Pandas, Scikit-Learn and
    Streamlit.

    The system predicts whether a loan
    application is likely to be approved
    based on applicant information.
    """)

    st.write("Developer Team:")

    st.write("• Kamlesh Pandey")
    st.write("• Prince Verma")
    st.write("• Nehal Kumar")