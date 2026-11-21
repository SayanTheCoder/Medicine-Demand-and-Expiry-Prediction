# 💊 Medicine Demand and Expiry Prediction

A Machine Learning-based web application developed using **Python and Streamlit** to predict future medicine demand and identify medicines approaching their expiry dates.

The project aims to help pharmacies and medical stores manage inventory efficiently, reduce medicine wastage, prevent stock shortages, and make better inventory management decisions.

## 📌 Project Overview

Managing medicine inventory is an important task for pharmacies. Overstocking can lead to medicine expiry and financial losses, while understocking can result in medicine shortages.

This project uses historical medicine demand data to predict next month's demand and applies date-based expiry monitoring to identify medicines approaching their expiry dates.

## 🎯 Objectives

- Predict the next month's demand for medicines using Machine Learning.
- Compare different regression models to identify the best-performing model.
- Monitor medicines approaching their expiry dates.
- Help pharmacies make better inventory management decisions.
- Reduce medicine wastage and prevent potential stock shortages.

## ✨ Features

### 1. Dashboard

- Overview of the medicine dataset.
- Total cleaned records.
- Total unique medicines.
- Forecast training records.

### 2. Data Overview

- Displays the complete cleaned dataset.
- Shows medicine details and historical demand information.

### 3. Medicine Demand Prediction

- Select a medicine by its name.
- Automatically displays the corresponding Medicine ID.
- Predicts the medicine's demand for the next month.
- Uses historical demand patterns for forecasting.

### 4. Machine Learning Models

The project uses three regression algorithms:

| Model                       | Purpose                                                            |
| --------------------------- | ------------------------------------------------------------------ |
| Decision Tree Regressor     | Predicts demand using decision-based rules                         |
| Multiple Linear Regression  | Predicts demand using multiple features                            |
| Gradient Boosting Regressor | Combines multiple decision trees to improve prediction performance |

### 5. Model Evaluation

The models are evaluated using three regression metrics:

| Metric   | Purpose                                               | Better Result |
| -------- | ----------------------------------------------------- | ------------- |
| MAE      | Measures average absolute prediction error            | Lower         |
| RMSE     | Penalizes larger prediction errors                    | Lower         |
| R² Score | Measures how well the model explains demand variation | Higher        |

### 6. Medicine Expiry Monitoring

- Select between This Month and Next Month.
- Generate an expiry report.
- Displays the number of medicines expiring in the selected month.
- Lists medicines approaching their expiry dates.
- Uses date-based logic to identify expiry information.

## 🛠️ Technologies Used

| Technology   | Purpose                            |
| ------------ | ---------------------------------- |
| Python       | Main programming language          |
| Pandas       | Data processing and manipulation   |
| NumPy        | Numerical operations               |
| Scikit-learn | Machine Learning model development |
| Matplotlib   | Data visualization                 |
| Seaborn      | Statistical visualization          |
| Streamlit    | Web application development        |

## 🤖 Machine Learning Approach

1. Prepare medicine inventory data.
2. Clean and preprocess the dataset.
3. Prepare forecasting features using historical records.
4. Create the next-month demand prediction target.
5. Split the data chronologically into training and testing sets.
6. Train three regression models.
7. Evaluate model performance using MAE, RMSE, and R² Score.
8. Select the best-performing model.
9. Predict the next month's medicine demand.

## 📊 Dataset Information

The project uses a synthetic medicine inventory and demand dataset.

| Dataset Information    | Count |
| ---------------------- | ----: |
| Cleaned Records        | 1,840 |
| Unique Medicines       |    40 |
| Forecast Training Rows | 1,800 |

### Dataset Attributes

- Record_Date
- Medicine_ID
- Medicine_Name
- Category
- Price
- Current_Stock
- Previous_Stock
- Quantity_Sold
- Quantity_Used
- Supplier_Lead_Days
- Expiry_Date
- Monthly_Demand

**Note:** The dataset is synthetic and created for academic and demonstration purposes. It does not represent actual pharmacy transactions.

## 📁 Project Structure

```text
Medicine_Demand_Expiry_Prediction/
│
├── medicine.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   │   └── medicine_demand_expiry_raw.csv
│   │
│   └── processed/
│       └── medicine_demand_expiry_cleaned.csv
│
└── src/
    └── clean_data.py
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/SayanTheCoder/Medicine-Demand-and-Expiry-Prediction.git
```

### 2. Navigate to the Project Folder

```bash
cd Medicine-Demand-and-Expiry-Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Data Cleaning Script

```bash
python src/clean_data.py
```

This processes the raw dataset and generates the cleaned dataset inside `data/processed/`.

### 7. Run the Streamlit Application

```bash
streamlit run medicine.py
```

The application will open in your default web browser.


## 📈 Expected Outcomes

- Better estimation of future medicine demand.
- Improved inventory planning.
- Early identification of medicines approaching expiry.
- Reduced medicine wastage.
- Better decision-making for pharmacy inventory management.

## 🔮 Future Improvements

- Integration with real-time pharmacy inventory databases.
- Use of real-world medicine sales datasets.
- Automated stock alerts and expiry notifications.
- Integration with pharmacy management systems.
- Improved forecasting using larger historical datasets.

## 👨‍💻 Developer

**Developed By: Sayan Pramanik**

## 🎓 Project Information

- **Project Name:** Medicine Demand and Expiry Prediction
- **Project Type:** Machine Learning Project
- **Application:** Web-Based Application
- **Domain:** Machine Learning and Healthcare Inventory Management
- **Developed Using:** Python and Streamlit

---

**Disclaimer:** This project is developed for academic and educational purposes. Its predictions should not be treated as professional medical or pharmaceutical advice.
