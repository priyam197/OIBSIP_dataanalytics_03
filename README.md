# OIBSIP_dataanalytics_03

# 📈 Task 3: House Price Prediction using Linear Regression

## **Internship**
**Oasis Infobyte** - Data Science Internship (Virtual)

## **Project Goal**
The primary objective of this project was to build a robust and interpretable predictive model for house prices using **Linear Regression**. The project involved a complete machine learning pipeline, focusing on data preprocessing, feature selection, and rigorous model evaluation.

## **Dataset**
The project utilized the `Housing.csv` dataset, containing various attributes related to residential properties (e.g., area, number of rooms, amenities, and price).

## **Key Steps & Methodology**

1.  **Data Cleaning & EDA:**
    * Performed **Outlier Treatment** on key variables like `price` and `area` to stabilize the target and predictor.
    * Visualized relationships using **Pair Plots** and **Violin Plots** (for categorical features vs. price).
2.  **Data Preparation & Feature Engineering:**
    * Converted binary categorical variables (`mainroad`, `airconditioning`, etc.) using **Binary Mapping (0/1)**.
    * Used **One-Hot Encoding (Dummy Variables)** for multi-level categorical features (`furnishingstatus`).
    * Applied **MinMax Scaling** to normalize all numerical features (including the target, `price`).
3.  **Model Building & Feature Selection:**
    * Split the data into Training (70%) and Testing (30%) sets.
    * Employed **Recursive Feature Elimination (RFE)** to select the optimal set of 6 features for the final model.
    * Built the model using `statsmodels` for detailed statistical analysis (P-values, Coefficients).
4.  **Model Validation:**
    * Checked for multicollinearity among predictors using the **Variance Inflation Factor (VIF)**.
    * Conducted **Residual Analysis** (plotting error term distribution).

## **Tools and Libraries**
* **Python**
* **Pandas & NumPy:** Data manipulation.
* **Matplotlib & Seaborn:** Visualization (Boxplots, Heatmaps).
* **Scikit-learn (sklearn):** `train_test_split`, `MinMaxScaler`, `LinearRegression`, `RFE`, `r2_score`.
* **Statsmodels:** For detailed regression statistics and VIF calculation.

## **Actionable Conclusions & Final Model**
* **Key Price Drivers:** The most significant factors identified by RFE are **Area, Bathrooms, Stories, Air Conditioning, Parking, and Preferred Area**.
* **Model Equation:** The final predictive equation provides clear insights into the impact of each variable:
    $$\text{price} = 0.35 \times \text{area} + 0.20 \times \text{bathrooms} + 0.19 \times \text{stories} + \ldots$$
* **Performance:** The model achieved a competitive $R^2$ score on the test set, demonstrating strong predictive power.

## **File Structure**
* `predicting_house_prices_with_linear_regression.py` - Primary Python script containing the entire ML pipeline and visualizations.
* `Housing.csv` - The original dataset file.
* `README.md` - This documentation file.
