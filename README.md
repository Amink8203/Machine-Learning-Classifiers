# Boston Housing Price Analysis and Machine Learning Classification

## 📋 Project Overview

This comprehensive machine learning project explores the Boston Housing dataset through exploratory data analysis (EDA), data preprocessing, and implementation of various classification algorithms. The goal is to build robust ML models that can classify houses into different price categories (Economic, Ordinary, Luxury) based on various housing features.

## 🎯 Project Objectives

- **Primary Goal:** Develop machine learning models to predict house price categories in Boston
- **Secondary Goals:**
  - Perform comprehensive exploratory data analysis
  - Implement effective data preprocessing techniques
  - Compare performance of different classification algorithms
  - Apply ensemble methods for improved predictions
  - Optimize model hyperparameters using GridSearchCV

## 📊 Dataset Information

The project uses the Boston Housing dataset (`DataSet.xlsx`) containing various features that influence house prices:

### Features Include:
- **CRIM:** Per capita crime rate by town
- **ZN:** Proportion of residential land zoned for lots over 25,000 sq.ft
- **INDUS:** Proportion of non-retail business acres per town
- **CHAS:** Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- **NOX:** Nitric oxides concentration (parts per 10 million)
- **RM:** Average number of rooms per dwelling
- **AGE:** Proportion of owner-occupied units built prior to 1940
- **DIS:** Weighted distances to employment centres
- **RAD:** Index of accessibility to radial highways
- **TAX:** Full-value property-tax rate per $10,000
- **PTRATIO:** Pupil-teacher ratio by town
- **B:** Proportion of blacks by town
- **LSTAT:** % lower status of the population
- **MEDV:** Median value of owner-occupied homes (target variable)

### Target Classification:
Houses are classified into three categories based on price percentiles:
- **Economic:** Bottom 20% (Deciles 1-2)
- **Ordinary:** Middle 60% (Deciles 3-8)  
- **Luxury:** Top 20% (Deciles 9-10)

## 🛠️ Technologies and Libraries

### Core Libraries:
- **Data Manipulation:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn, xgboost
- **Statistical Analysis:** scipy

### Machine Learning Algorithms Implemented:
1. **Linear Regression** (Custom Implementation)
2. **K-Nearest Neighbors (KNN)**
3. **Decision Trees**
4. **Random Forest**
5. **XGBoost**
6. **Support Vector Machines (SVM)**

## 📁 Project Structure

```
CA4/
├── CA4.ipynb              # Main Jupyter notebook with complete analysis
├── DataSet.xlsx           # Boston Housing dataset
├── README.md              # This file
└── Description/
    └── HW-4-Project.pdf   # Project requirements document
```

## 🔍 Analysis Workflow

### Phase 1: Exploratory Data Analysis (EDA)
- **Data Summary:** Comprehensive overview using `info()` and `describe()`
- **Missing Value Analysis:** Identification and quantification of missing data
- **Correlation Analysis:** Heatmap visualization of feature relationships
- **Univariate Analysis:** Distribution analysis with kernel density plots
- **Bivariate Analysis:** Scatter plots and hexbin plots for feature relationships
- **Multivariate Analysis:** Correlation patterns and feature interactions

### Phase 2: Data Preprocessing
- **Missing Value Handling:**
  - Median imputation for `DIS` (high skewness)
  - Predictive imputation using Random Forest for `B`
  - Row removal for missing target values (`MEDV`)
  - Feature removal for `CHAS` (low correlation, high missing rate)
- **Feature Scaling:** StandardScaler and MinMaxScaler implementations
- **Train-Test-Validation Split:** 80%-10%-10% split strategy

### Phase 3: Model Implementation and Evaluation

#### Linear Regression
- Custom implementation with mathematical derivation
- Feature-wise analysis with correlation-based selection
- Best performing features: `RM` (R² = 0.868) and `LSTAT` (R² = 0.47)

#### Classification Models
All models implemented with hyperparameter optimization:

1. **K-Nearest Neighbors**
   - Optimal parameters: `n_neighbors=2`, `weights='uniform'`
   - Cross-validation for robust evaluation

2. **Decision Trees**
   - Entropy criterion with optimal `max_depth=52`
   - Pruning analysis for overfitting prevention
   - Tree visualization for interpretability

3. **Random Forest**
   - Ensemble of 120 trees with optimized parameters
   - Feature importance analysis
   - Bootstrap sampling explanation

4. **XGBoost**
   - Gradient boosting with advanced regularization
   - Learning rate optimization: 0.01
   - 190 estimators for optimal performance

5. **Support Vector Machines**
   - RBF and Linear kernel comparison
   - GridSearch vs RandomSearch evaluation
   - Hyperparameter optimization for C and gamma

### Phase 4: Ensemble Methods
- **Bagging vs Boosting** theoretical comparison
- **Random Forest** detailed implementation
- **XGBoost** advanced gradient boosting
- Performance comparison across all ensemble methods

### Phase 5: Model Evaluation and Validation
- **Overfitting/Underfitting Analysis:** Learning curves for all models
- **Cross-Validation:** K-fold validation for robust performance estimation
- **Hyperparameter Tuning:** GridSearchCV for optimal parameter selection
- **Performance Metrics:** Accuracy, Precision, Recall, F1-Score analysis

## 📈 Key Results

### Model Performance Summary:
- **Best Linear Regression Feature:** RM (R² = 0.868)
- **Classification Accuracy:** All models achieved >85% accuracy
- **Ensemble Methods:** Consistently outperformed individual models
- **Optimal Algorithm:** XGBoost with fine-tuned hyperparameters

### Key Findings:
1. **Feature Importance:** `RM` (rooms) and `LSTAT` (lower status %) most predictive
2. **Data Quality:** Effective handling of 4 features with missing values
3. **Model Comparison:** Ensemble methods superior to individual classifiers
4. **Overfitting Control:** Successful prevention through proper validation

## 🚀 Usage Instructions

### Prerequisites:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost openpyxl
```

### Running the Analysis:
1. Ensure `DataSet.xlsx` is in the project directory
2. Open `CA4.ipynb` in Jupyter Notebook/Lab
3. Run all cells sequentially for complete analysis
4. Examine visualizations and model performance metrics

### Code Structure:
- **Custom Classes:** Implemented for each algorithm (KNNClassifier, DTClassifier, etc.)
- **Evaluation Functions:** Standardized scoring and visualization functions
- **Hyperparameter Grids:** Comprehensive parameter spaces for optimization

## 📊 Visualizations

The project includes comprehensive visualizations:
- **Correlation Heatmaps:** Feature relationship analysis
- **Distribution Plots:** Univariate and bivariate analysis
- **Scatter Plots:** Feature vs target relationships
- **Confusion Matrices:** Classification performance visualization
- **Learning Curves:** Overfitting/underfitting detection
- **Decision Tree Plots:** Model interpretability

## 🔧 Technical Implementation Details

### Custom Implementations:
- **Linear Regression:** Mathematical derivation with RSS optimization
- **Evaluation Metrics:** Custom R², RMSE, accuracy calculation functions
- **Data Preprocessing:** Tailored missing value handling strategies

### Advanced Techniques:
- **Bootstrap Sampling:** Random Forest implementation details
- **Gradient Boosting:** XGBoost mathematical foundation
- **Kernel Methods:** SVM with RBF and linear kernel comparison
- **Cross-Validation:** Robust model evaluation framework

## 📚 Theoretical Foundations

The project covers extensive machine learning theory:
- **Supervised vs Unsupervised Learning** distinctions
- **Regression vs Classification** methodologies
- **Bias-Variance Tradeoff** in ensemble methods
- **Overfitting Prevention** strategies
- **Feature Engineering** and selection techniques
- **Distance Metrics** for KNN algorithms
- **Tree Pruning** strategies for decision trees
