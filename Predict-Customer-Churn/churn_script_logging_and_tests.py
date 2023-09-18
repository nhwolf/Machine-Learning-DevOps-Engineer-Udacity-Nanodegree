'''
Module contains tests for customer churn analysis functions in
churn_library.py

Author: Nicholas Wolf

Last Modified: September 17, 2023
'''

import os
import logging
import constants
import churn_library as churn_lib

logging.basicConfig(
	filename='./logs/churn_library.log',
	level = logging.INFO,
	filemode='w',
	format='%(name)s - %(levelname)s - %(message)s')

def test_import():
    '''
	test data import - this example is completed for you to assist with the other test functions
	'''
    try:
        df = churn_lib.import_data(constants.DATA_PTH)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    Test perform_eda() function from the churn_library module
    '''
    df = churn_lib.import_data(constants.DATA_PTH)
    try:
        churn_lib.perform_eda(df)
        logging.info("Testing perform_eda: SUCCESS")
    except KeyError as err:
        
       
        logging.error('Column "%s" not found', err.args[0])
        raise err

    # Assert if Churn Histogram is created
    try:
        assert os.path.isfile(constants.EDA_CHURN_HIST_PATH) is True
        logging.info('File %s was found', 'churn_hist.png')
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err

    # Assert if Customer Age Histogram is created
    try:
        assert os.path.isfile(constants.EDA_CUST_AGE_HIST_PATH) is True
        logging.info('File %s was found', 'customer_age_hist.png')
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err

    # Assert if Marital Status Histogram is created
    try:
        assert os.path.isfile(constants.EDA_MARITAL_STATUS_HIST_PATH) is True
        logging.info('File %s was found', 'marital_status_hist.png')
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err

    # Assert if Total Transaction Distribution Histogram is created
    try:
        assert os.path.isfile(constants.EDA_DIST_PLOT_PATH ) is True
        logging.info('File %s was found', 'dist_plot.png')
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err

    # Assert if Heatmap is created
    try:
        assert os.path.isfile(constants.EDA_HEATMAP_PATH) is True
        logging.info('File %s was found', 'heatmap.png')
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err
      

def test_encoder_helper():
    '''
    Test encoder_helper() function from the churn_library module
    '''
    # Load the dataframe
    df = churn_lib.import_data(constants.DATA_PTH)

    # Categorical Features
    cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
                   'Income_Category', 'Card_Category']

    try:
        encoded_df = churn_lib.encoder_helper(
                            df=df,
                            category_lst=[],
                            response=None)

        # Data should be the same
        assert encoded_df.equals(df) is True
        logging.info("Testing encoder_helper(df, category_lst=[]): SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper(df, category_lst=[]): ERROR")
        raise err

    try:
        encoded_df = churn_lib.encoder_helper(
                            df=df,
                            category_lst=cat_columns,
                            response=None)

        # Checking column names
        assert encoded_df.columns.equals(df.columns) is True

        # Checking data contents
        assert encoded_df.equals(df) is False
        logging.info(
            "Testing encoder_helper(df, category_lst=cat_columns, response=None): SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper(df, category_lst=cat_columns, response=None): ERROR")
        raise err

    try:
        encoded_df = churn_lib.encoder_helper(
                            df=df,
                            category_lst=cat_columns,
                            response='Churn')

        # Checking column names
        assert encoded_df.columns.equals(df.columns) is False   

        # Checking data
        assert encoded_df.equals(df) is False

        # Num. columns in encoded_df = columns in df + newly created columns from cat_columns
        assert len(encoded_df.columns) == len(df.columns) + len(cat_columns)    
        logging.info(
        "Testing encoder_helper(df, category_lst=cat_columns, response='Churn'): SUCCESS")
    except AssertionError as err:
        logging.error(
        "Testing encoder_helper(df, category_lst=cat_columns, response='Churn'): ERROR")
        raise err


def test_perform_feature_engineering():
	'''
	test perform_feature_engineering
	'''

    # Load the dataframe
	df = churn_lib.import_data(constants.DATA_PTH)

	try:
		(_, X_test, _, _) = churn_lib.perform_feature_engineering(
            df=df,
            response='Churn'
        )
        # Checking that 'Churn' exists in the df dataframe
		assert 'Churn' in df.columns
		logging.info("Testing perform_feature_engineering. 'Churn' column is present: SUCCESS")
	except KeyError as err:
		logging.error('The "Churn" column is not present in the data frame: ERROR')
		raise err



def test_train_models():
	'''
	test train_models
	'''
        # Load the DataFrame
	df = churn_lib.import_data(constants.DATA_PTH)



    # Feature engineering 
	(X_train, X_test, y_train, y_test) = clib.perform_feature_engineering(  
                                                    dataframe=dataframe,
                                                    response='Churn')

    # Assert if `logistic_model.pkl` file is present
	try:
		churn_lib.train_models(X_train, X_test, y_train, y_test)
		assert os.path.isfile(constants.RESULTS_LOGISTIC_REGRESSION_PATH) is True
		logging.info('File %s was found', 'logistic_model.pkl')
	except AssertionError as err:
		logging.error('Not such file on disk')
		raise err

    # Assert if `rfc_model.pkl` file is present
	try:
		assert os.path.isfile(constants.MODELS_RFC_PATH) is True
		logging.info('File %s was found', 'rfc_model.pkl')
	except AssertionError as err:
		logging.error('Not such file on disk')
		raise err

    # Assert if `rfc_results.png` file is present
	try:
		assert os.path.isfile(constants.RESULTS_RANDOM_FOREST_PATH) is True
		logging.info('File %s was found', 'random_forest_results.png')
	except AssertionError as err:
		logging.error('Not such file on disk')
		raise err

    # Assert if `logistic_results.png` file is present
	try:
		assert os.path.isfile(constants.RESULTS_LOGISTIC_REGRESSION_PATH) is True
		logging.info('File %s was found', 'logistic_regression_results.png')
	except AssertionError as err:
		logging.error('Not such file on disk')
		raise err

    # Assert if `feature_importances.png` file is present
	try:
		assert os.path.isfile('./images/results/feature_importances.png') is True
		logging.info('File %s was found', 'feature_importances.png')
	except AssertionError as err:
		logging.error('Not such file on disk')
		raise err


if __name__ == "__main__":
	pass