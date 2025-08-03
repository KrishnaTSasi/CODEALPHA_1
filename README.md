# CODEALPHA_1

 Iris Flower Classification - CODSOFT Task 1

This project classifies iris flowers into three species using machine learning models.

 ## Dataset
- Source: sklearn.datasets.load_iris()
- Features: Sepal length, Sepal width, Petal length, Petal width
- Classes: Setosa, Versicolor, Virginica

 ## Models Used

| Model                  | Accuracy |
|------------------------|----------|
| Logistic Regression    | 1.00     |
| Decision Tree          | 1.00     |
| Random Forest          | 1.00     |
| K-Nearest Neighbors    | 1.00     |
| Gaussian Naive Bayes   | 1.00     |
| Support Vector Machine | 0.97     |
| MLP Classifier (NN)    | 1.00     |


 ## Visualizations

- Pair plot of features
- Heatmap of correlations
- Boxplots and violin plots per species
- Confusion matrices for each model

##  Step 1: Import Required Libraries

* scikit-learn for models and metrics
* pandas, numpy for data handling
* seaborn, matplotlib for visualizations

##  Step 2: Load the Dataset

* Use load_iris() from sklearn.datasets
* Extract X (features) and y (labels)
* Target classes: Setosa, Versicolor, Virginica

##  Step 3: Data Exploration & Visualization

* Create pair plots, box plots and correlation heatmaps
* Understand feature distribution and class separation
* Check for class imbalance (Iris is balanced)

## Step 4: Preprocessing

* Train-Test Split: 70% train, 30% test using train_test_split
* Feature Scaling: Standardize features using StandardScaler

##  Step 5: Train Multiple Models

Train and evaluate the following classifiers:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. Gaussian Naive Bayes
6. Support Vector Machine (SVM)
7. MLP Classifier (Neural Network)

## Step 6: Evaluate Models

For each model:

* Calculate Accuracy
* Print Classification Report
* Plot Confusion Matrix using seaborn.heatmap


##  Author

Krishna T Sasi – CODSOFT Internship – Task 1



