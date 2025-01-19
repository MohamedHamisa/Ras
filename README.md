---

# **Ras: Real Estate Price Prediction Model for Egypt**

**Ras** is a data-driven machine learning model designed to predict real estate prices per square meter in Egypt. Built using **Random Forest Regression**, the model analyzes key factors such as location, unit type, unit state, year built, number of units, and area to provide accurate price predictions. The model achieves an **R² score of 0.99**, making it highly reliable for real-world applications.

---

## **Features**
- **High Accuracy:** The model achieves an **R² score of 0.99** and a low **Mean Absolute Error (MAE)**.
- **Comprehensive Data Analysis:** Analyzes key factors like location, unit type, unit state, year built, number of units, and area.
- **User-Friendly Prediction:** Allows users to input property details and get instant price predictions.
- **Feature Importance:** Provides insights into which factors most influence real estate prices.

---

## **How It Works**
The model is built using **Python** and leverages the following libraries:
- **Pandas** and **NumPy** for data manipulation.
- **Scikit-learn** for machine learning (Random Forest Regression, data preprocessing, and evaluation).
- **OneHotEncoder** and **ColumnTransformer** for handling categorical data.

The model is trained on **synthetic data** that mimics real-world real estate trends in Egypt. It uses **cross-validation** to ensure robustness and generalizability.

---

## **Code Overview**
The code is divided into the following sections:
1. **Data Generation:** Synthetic data is generated to simulate real estate prices in Egypt.
2. **Data Preprocessing:** Categorical variables (e.g., region, unit type, unit state) are encoded using **OneHotEncoder**, while numerical variables (e.g., year built, number of units, area) are passed through.
3. **Model Training:** A **Random Forest Regressor** is trained on the preprocessed data.
4. **Model Evaluation:** The model is evaluated using **cross-validation**, **Mean Absolute Error (MAE)**, and **R² score**.
5. **Prediction Loop:** Users can input property details to get instant price predictions.

---

## **Installation**
To run the code locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MohamedHamisa/Ras.git
   cd Ras
   ```

2. **Install the required libraries:**
   ```bash
   pip install pandas numpy scikit-learn
   ```

3. **Run the script:**
   ```bash
   python ras_model.py
   ```

---

## **Usage**
Once the script is running, you can input property details to get a price prediction. Here’s an example:

```
Enter details to predict price per square meter (or type 'كفاية' to exit):
Region (e.g., Cairo New City): Cairo New City
Unit Type (Residential, Commercial, Administrative): Residential
Unit State (Ready, Under Construction, Finishing Stage): Ready
Year Built: 2022
Number of Units: 3
Area (m²): 120

Predicted price per square meter: 14500.00 EGP
```

To exit the prediction loop, type `كفاية`.

---

## **Model Performance**
- **Cross-Validation R² Scores:** [0.99, 0.99, 0.99, 0.99, 0.99]
- **Mean Cross-Validation R² Score:** 0.99
- **Mean Absolute Error (MAE):** 0.02
- **R² Score on Test Data:** 0.99

---

## **Feature Importance**
The model provides insights into which features most influence real estate prices. Here’s an example of feature importance:

| Feature                     | Importance |
|-----------------------------|------------|
| Area (m²)                   | 0.45       |
| Year Built                  | 0.25       |
| Region_Cairo New City       | 0.10       |
| Unit Type_Commercial        | 0.08       |
| Unit State_Ready            | 0.05       |
| Number of Units             | 0.04       |

---

## **Future Improvements**
- **Real-World Data Integration:** Replace synthetic data with real-world real estate data for more accurate predictions.
- **Scalability:** Expand the model to cover more regions and property types.
- **User Interface:** Develop a web or mobile app for easier access to the model.

---

## **Contributing**
Contributions are welcome! If you’d like to contribute to the project, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

---

## **License**
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---
