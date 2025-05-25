House Prices Project

This project is an end-to-end Machine Learning pipeline designed to predict house prices based on a public dataset. Our journey includes meticulous data preparation, with a special focus on managing multicollinearity among features using an iterative method based on the Variance Inflation Factor (VIF). Ultimately, we'll compare the performance of three popular linear regression models: standard Linear Regression, Ridge, and Lasso.
Dataset

The core of our project is the house_data.csv (or kc_house_data.csv) dataset. It holds a wealth of information about real estate properties to help us uncover the secrets of house pricing!
Project Objectives

Here's what we aim to do:

    Load and explore the data to understand its characteristics.
    Clean the data, saying goodbye to duplicates.
    Analyze the correlation between features with a neat heatmap.
    Tackle multicollinearity: identify and mitigate problematic features using an iterative removal process based on VIF.
    Standardize features to best prepare them for regularized models.
    Split the data into training and testing sets for fair training and evaluation.
    Train our three regression models: Linear Regression, Ridge, and Lasso.
    Evaluate each model's performance using appropriate regression metrics (R2 Score, MSE, RMSE, MAE).

Project Structure

To keep everything tidy, the code is split into files with specific responsibilities:

    house.py: The brain behind the operation. It loads the data, calls the pre-processing functions, and finally trains/evaluates the models.
    preprocess.py: The cleaner and VIF selector. It contains the preprocess(df) function for initial cleaning and iterative VIF management.
    features.py: The standardizer and splitter. It contains the feature(X_selected, y) function for scaling and train/test splitting on the already filtered data.

Requirements

To run this project on your PC, make sure you have the following installed:

    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    statsmodels

The easiest way to install them is via pip in a virtual environment:
Bash

    pip install pandas numpy matplotlib seaborn scikit-learn statsmodels

If you have a requirements.txt file, you can simply use:
Bash

    pip install -r requirements.txt

How to Run the Script

Follow these simple steps:

    Clone (or download) this repository to your computer.

    Ensure that the dataset file (house_data.csv or kc_house_data.csv) is in the same folder as the Python scripts, or update the path in the code.

    Open your terminal or command prompt in the project directory.

    Run the main script:
    Bash

    python house.py

Model Evaluation

Each model is trained on the standardized training set and evaluated on the test set using fundamental regression metrics:

    R2 Score (how much of the price variation is explained by the model)
    Mean Squared Error (MSE)
    Root Mean Squared Error (RMSE) – typical error in the same unit as the price
    Mean Absolute Error (MAE)

These numbers tell us how well each model predicts prices on unseen data.
Results and Visualizations (Optional)

The script will print the evaluation metrics for each model. For a deeper visual analysis, you might add plots such as:

    A scatter plot of actual prices vs. predicted prices (ideally: points on a perfect diagonal line).
    A plot of residuals to check for model error.

Authors: Giacomo Visciotti, Simone Verrengia
