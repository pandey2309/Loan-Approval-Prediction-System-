# 🏦 Loan Approval Prediction System

This is a Machine Learning-based web application that predicts whether a loan applicant will be approved by a bank or not. The project is built using Python and Streamlit.

## 👥 Project Team
* **Kamlesh Pandey**
* **Prince Verma**
* **Nehal Kumar**

## 🌟 Features of this App
Our Streamlit application includes the following pages and features:
* **🏠 Home:** Basic overview of the project and model accuracy details.
* **📊 Dataset Overview:** Displays the first 10 records of the loan dataset and its statistical summary.
* **📈 Data Visualizations:** Includes pie charts and bar graphs showing loan status, education, and property area distributions.
* **🤖 Model Performance:** A comparison of different Machine Learning models (Logistic Regression, Decision Tree, Random Forest).
* **📝 Loan Prediction:** An interactive form where users can input details (Income, Education, Loan Amount, etc.) to check real-time loan approval status.
* **ℹ️ About Project:** Brief information about the project and the developer team.

## 🛠️ Tech Stack Used
* **Language:** Python
* **Web Framework:** Streamlit
* **Machine Learning:** Scikit-Learn
* **Data Processing:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Model Loading:** Joblib

## 🚀 How to Run the Project Locally

Follow these steps to run this project on your local machine:

**Step 1: Clone the repository**
```bash
git clone <YOUR_REPO_URL_HERE>
Step 2: Navigate to the project directory

Bash
cd <YOUR_REPO_NAME>
Step 3: Install the required libraries

Bash
pip install -r requirements.txt
(Alternatively, you can install them directly: pip install pandas numpy scikit-learn streamlit matplotlib seaborn joblib)

Step 4: Run the application

Bash
streamlit run app.py
📁 Folder Structure
To ensure the app runs without errors, make sure your folder structure looks like this:

📂 Project_Folder
 ┣ 📂 dataset
 ┃ ┗ 📜 loan_approval_dataset.csv   <-- Your dataset file
 ┣ 📂 models
 ┃ ┗ 📜 loan_model.pkl              <-- Your trained ML model
 ┣ 📜 app.py                        <-- Streamlit main code file
 ┣ 📜 requirements.txt              <-- List of required libraries
 ┗ 📜 README.md                     <-- This file
