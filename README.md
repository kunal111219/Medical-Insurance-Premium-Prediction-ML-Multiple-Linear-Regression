# Insurance Charges Prediction using Linear Regression

This project uses a machine learning model to predict insurance charges based on personal and medical information of individuals, such as age, gender, BMI, number of children, smoking habits, and geographical region. The model is built using **Linear Regression** and evaluated with metrics such as **R-squared (R²)**.

## Overview
This project demonstrates how to build a predictive model for insurance charges using a dataset that contains various demographic and health-related features. After performing Exploratory Data Analysis (EDA) and data preprocessing, a Linear Regression model is trained to predict insurance costs.

## Technologies
- **Python**: 3.x
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Seaborn & Matplotlib**: Data visualization
- **Scikit-learn**: Machine learning library for training and evaluating the model

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/kunal111219/Medical-Insurance-Premium-Prediction-ML-Multiple-Linear-Regression.git 
    cd Medical-Insurance-Premium-Prediction-ML-Multiple-Linear-Regression
    ```

2. Install the required dependencies:
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn
    ```

3. Run the script:
    ```bash
    python Medical_Insurance_Cost_Prediction.ipynb
    ```

## Dataset

The dataset used for this project is the **Insurance Dataset**, which contains the following columns:

- **insuree_id**: Insurence ID 
- **age**: Age of the individual
- **sex**: Gender of the individual (male/female)
- **bmi**: Body Mass Index (BMI)
- **children**: Number of children covered by the insurance
- **smoker**: Whether the individual is a smoker (yes/no)
- **region**: Geographical region (northeast, northwest, southeast, southwest)
- **charges**: Medical insurance cost (target variable)

## Exploratory Data Analysis (EDA)

The dataset was explored using visualizations like histograms and count plots. Key distributions and relationships between variables such as **age**, **bmi**, **smoker**, and **charges** were analyzed.

## Model Building

We used **Linear Regression** as the prediction model. Key steps include:
- Encoding categorical variables like **sex**, **smoker**, and **region**.
- Splitting the data into training and test sets (80/20 split).
- Fitting the **Linear Regression** model using **Scikit-learn**.

## Evaluation

The model's performance is evaluated using the **R² score**:
- **Training R² Score**: ~0.751
- **Testing R² Score**: ~0.744

## Usage

After training the model, you can use it to make predictions on new data by providing the following inputs:

```python
# Example input: (age, sex, bmi, children, smoker, region)
input_data = (31, 1, 25.74, 0, 1, 0)

# Prediction for the input data
The insurance cost is \$ 3760.68
```

## Contributing

Feel free to contribute to this project by submitting issues or pull requests. For major changes, please open an issue first to discuss what you want to change.\

## Contact

For any questions or feedback, please reach out to <rastogikunal19@gmail.com>.
