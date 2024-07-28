<h1 align="center">Data Science & Machine Learning : Housing Price Prediction Project<p align="center"></h1>

<h1 align="center">Bar Bettash<p align="center">
<a href="https://www.linkedin.com/in/barbettash/" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="LinkedIn" height="30" width="40" /></a>
</h1>


<!-- ABOUT THE PROJECT -->
## About The Project

This project aims to predict housing prices using a dataset containing various features related to housing in California. The project includes data cleaning, exploratory data analysis (EDA), feature engineering, data visualization, and model training using linear regression and random forest regression, with hyperparameter tuning using grid search.

### Data from
https://www.kaggle.com/datasets/camnugent/california-housing-prices

### Built With

![image](https://github.com/user-attachments/assets/7e3ed475-d755-480f-b808-6331248bd409)



Python <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a>

![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/> </a>

![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)  <a href="https://numpy.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/numpy/numpy-original.svg" alt="numpy" width="40" height="40"/> </a>

![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) <a href="https://matplotlib.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/matplotlib/matplotlib-original.svg" alt="matplotlib" width="40" height="40"/> </a>

![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) </p>


<!-- GETTING STARTED -->
## Getting Started


### Prerequisites

To run the project, you need the following libraries installed:

* pandas: A powerful library for data manipulation, analysis, and loading data from various sources (CSV in this case).
  ```sh
  pip install pandas

* numpy: Provides efficient operations on numerical data (arrays, matrices) essential for machine learning algorithms.
  ```sh
  pip install numpy

* matplotlib: Used for creating static visualizations like histograms and scatterplots to explore data.
  ```sh
  pip install matplotlib

* seaborn: A high-level library built on top of matplotlib for creating more sophisticated and visually appealing data visualizations.
  ```sh
  pip install seaborn

* scikit-learn: For machine learning algorithms. Offers functions for splitting datasets into training and testing sets, a crucial step in machine learning.
  ```sh
  pip install scikit-learn


### Load dataset:

The housing dataset is loaded into a Pandas DataFrame for further analysis.

### Data exploration: 

Basic exploration is performed using data and data.info() to understand the data structure and column information.

## Data Preprocessing:

### Handle missing values: 

Rows with missing values are dropped using dropna().

### Feature selection: 

The target variable ('median_house_value') is separated from the features (X), forming the basis for the model.

## Data Splitting:

### Train-test split: 
The dataset is divided into training and testing sets using train_test_split. A 20% portion is allocated for testing.

## Feature Engineering:

### Log transformation:

The total_rooms, total_bedrooms, population, and households features are log-transformed to potentially improve model performance and address skewed distributions.

### One-hot encoding: 
The categorical 'ocean_proximity' feature is converted into numerical dummy variables using pd.get_dummies.

### Feature creation: 

New features like bedroom_ratio and household_rooms are derived to capture potential relationships between variables.

## Exploratory Data Analysis (EDA):

### Histograms: 
Visualize the distribution of numerical features to identify potential outliers or skewness.
![image](https://github.com/user-attachments/assets/ea24922a-d982-4891-8138-86e32710f36a)


### Correlation matrix: 

Analyze the correlation between numerical features using a heatmap.
![image](https://github.com/user-attachments/assets/32a95ba8-7780-4112-b95e-79880be6e48e)


### Scatterplot: 

Explore the relationship between latitude, longitude, and median house value using a scatterplot with color-coded values.
![image](https://github.com/user-attachments/assets/41607d3d-b883-43c6-a136-970006bdf404)


## Model Training:

### Linear regression: 

A LinearRegression model is instantiated and fitted on the training data.

### Model evaluation: 
The model's performance is assessed on the testing set using the score method, which calculates the R^2 score.

## Scaling and Model Optimization:

### Scaling: 
StandardScaler is applied to standardize features for better model performance.

### Cross-validation: 
GridSearchCV is used to optimize hyperparameters (n_estimators, min_samples_split, max_depth) for a RandomForestRegressor.



### Example:
--------------------------------------------------

The best forest model's accuracy is 0.8094614517145665

--------------------------------------------------

<!-- CONTACT -->
## Contact

<p align="left">
<a href="https://www.linkedin.com/in/barbettash/" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="https://www.linkedin.com/in/barbettash/" height="30" width="40" /></a>
</p>


**bar.bettash.jobs@gmail.com** 


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This code is provided "as is" and may not function as intended in the future due to potential updates to external libraries, frameworks, websites, or APIs it interacts with. The code is no longer actively maintained and may require modifications to adapt to future changes.

**Recommendations:**

* Keep an eye on updates to libraries and dependencies to ensure compatibility.
* Be prepared to adapt the code based on future changes in the target website or API.








  
