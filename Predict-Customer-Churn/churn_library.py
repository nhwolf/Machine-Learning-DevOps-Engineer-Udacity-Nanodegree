'''
churn_library.py is a library of functions to find customers who are likely to churn

Author: Nicholas Wolf

Last Modified: September 17, 2023
'''


# import libraries
#from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import os
import shap
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import constants
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    
    df =  pd.read_csv(pth)   
    
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val=="Existing Customer" else 1)
    
    df['Churn'] = pd.to_numeric(df['Churn'])
    
    return df


def perform_eda(df):
    '''
    performs EDA on the df and save figures to images folder

    input:
            df: pandas dataframe

    output:
            None
    '''

    # Copy DataFrame
    eda_df = df
    
    # Churn Histogram
    plt.figure(figsize=(20, 10))
    eda_df['Churn'].hist()
    plt.savefig(constants.EDA_CHURN_HIST_PATH)

    # Customer Age Histogram
    eda_df['Customer_Age'].hist()
    plt.savefig(constants.EDA_CUST_AGE_HIST_PATH)

    # Marital Status Histogram
    plt.figure(figsize=(20, 10))
    eda_df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(constants.EDA_MARITAL_STATUS_HIST_PATH)

    # Total Transaction Distribution Histogram
    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(constants.EDA_DIST_PLOT_PATH)

    # Heatmap
    # Heatmap (excluding categorical columns)
    numerical_df = eda_df.select_dtypes(include=['number'])  #only numerical columns
    plt.figure(figsize=(20, 10))
    sns.heatmap(numerical_df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(constants.EDA_HEATMAP_PATH)


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used
                      for naming variables or index y column]

    output:
            df: pandas dataframe with new columns
    '''
    # Copy DataFrmae
    encoder_df = df

    for category in category_lst:
        column_lst = []
        column_groups = df.groupby(category).mean()['Churn']

        for val in df[category]:
            column_lst.append(column_groups.loc[val])

        if response:
            encoder_df[category + '_' + response] = column_lst
        else:
            encoder_df[category] = column_lst

    return encoder_df


def perform_feature_engineering(df, response=None):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could
                        be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    # Categorical features
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    # Feature engineering
    encoded_df = encoder_helper(
        df=df,
        category_lst=cat_columns,
        response=response)

    # Target feature
    y = encoded_df['Churn']

    # Creating dataframe
    X = pd.DataFrame()

    # Columns to keep
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    X[keep_cols] = encoded_df[keep_cols]

    # Splitting into train and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return (X_train, X_test, y_train, y_test)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # Random Forest
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25,
             str('Random Forest Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05,
             str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6,
             str('Random Forest Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7,
             str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(constants.RESULTS_RANDOM_FOREST_PATH)

    # Logistic Regression
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25,
             str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05,
             str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6,
             str('Logistic Regression Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7,
             str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(constants.RESULTS_LOGISTIC_REGRESSION_PATH)


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Sort feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save the image
    plt.savefig(fname=output_pth + 'feature_importances.png')


def shap_explainer_plot(model, X_data, output_pth):
    '''

    Generate a SHAP summary plot for a given model's feature importance using SHAP values.

    Inputs:
        model: model for which SHAP values will be calculated.
        X_data (array-like): The input data on which to compute SHAP values.
        output_pth: The file path where the SHAP summary plot will be saved.

    Outpu: None
    '''
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    shap.summary_plot(shap_values, X_data, plot_type="bar")
    plt.savefig(output_pth)


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # Random Forest Classifier and Logistic Regression
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    # Parameters for Grid Search
    param_grid = {'n_estimators': [200, 500],
                  'max_features': ['auto', 'sqrt'],
                  'max_depth': [4, 5, 100],
                  'criterion': ['gini', 'entropy']}

    # Grid Search and fit for RandomForestClassifier
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # LogisticRegression
    lrc.fit(X_train, y_train)

    # Save best models
    joblib.dump(cv_rfc.best_estimator_, constants.MODELS_RFC_PATH)
    joblib.dump(lrc, constants.MODELS_LOGISTIC_PATH)

    # Compute train and test predictions for RandomForestClassifier
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    # Compute train and test predictions for LogisticRegression
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Compute ROC curve
    # plot_roc_curve was removed from scikit-learn in v1.2
    # Need to fix by using sklearn.metrics.RocCurveDisplay
    # plt.figure(figsize=(15, 8))
    # axis = plt.gca()
    # lrc_plot = plot_roc_curve(lrc, X_test, y_test, ax=axis, alpha=0.8)
    # rfc_disp = plot_roc_curve(
    # cv_rfc.best_estimator_,
    # X_test,
    # y_test,
    # ax=axis,
    # alpha=0.8)
    # plt.savefig(fname='./images/results/roc_curve_result.png')

    # Compute and results
    classification_report_image(y_train, y_test,
                                y_train_preds_lr, y_train_preds_rf,
                                y_test_preds_lr, y_test_preds_rf)

    # Shap Explainer Plot
    shap_explainer_plot(
        cv_rfc.best_estimator_,
        X_test,
        constants.RESULTS_SHAP_EXPLAINER_PATH)

    # Compute and feature importance
    feature_importance_plot(model=cv_rfc,
                            X_data=X_test,
                            output_pth='./images/results/')


def main():
    '''
    Main function for the churn_library pipeline.

    This function performs the following steps:
    1. Imports data from the './data/bank_data.csv' file using the 'import_data' function.
    2. Conducts Exploratory Data Analysis (EDA) using the 'perform_eda' function.
    3. Performs feature engineering on the EDA results using the 
       'perform_feature_engineering' function.
    4. Trains machine learning models using the 'train_models' function.
    
    input:
              None

    output:
              None
    '''
    # Importing data
    df = import_data(constants.DATA_PTH)

    # Perform EDA
    eda_df = perform_eda(df)

    # Feature Engineering
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df=eda_df,response='Churn')

    # Model training
    train_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
