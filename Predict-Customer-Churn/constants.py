'''
constants.py stores constants required in the churn_library.py file

Author: Nicholas Wolf

Last Modified: September 17, 2023
'''

#Input data
DATA_PTH = './data/bank_data.csv'

#Images - EDA
EDA_CHURN_HIST_PATH = './images/eda/churn_hist.png'
EDA_CUST_AGE_HIST_PATH = './images/eda/customer_age_hist.png'
EDA_MARITAL_STATUS_HIST_PATH = './images/eda/marital_status_hist.png'
EDA_DIST_PLOT_PATH = './images/eda/dist_plot.png'
EDA_HEATMAP_PATH = './images/eda/heatmap.png'

#Images - Results
RESULTS_RANDOM_FOREST_PATH = './images/results/random_forest_results.png'
RESULTS_LOGISTIC_REGRESSION_PATH = './images/results/logistic_regression_results.png'
RESULTS_SHAP_EXPLAINER_PATH = './images/results/rf_shap_values_summary.png'

#Models
MODELS_RFC_PATH = './models/rfc_model.pkl'
MODELS_LOGISTIC_PATH = './models/logistic_model.pkl'
