'''
salary_predictor.py
Predictor of salary from old census data -- riveting!
'''
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

class SalaryPredictor:

    def __init__(self, X_train, y_train):
        """
        Creates a new SalaryPredictor trained on the given features from the
        preprocessed census data to predicted salary labels. Performs and fits
        any preprocessing methods (e.g., imputing of missing features,
        discretization of continuous variables, etc.) on the inputs, and saves
        these as attributes to later transform test inputs.
        
        :param DataFrame X_train: Pandas DataFrame consisting of the
        sample rows of attributes pertaining to each individual
        :param DataFrame y_train: Pandas DataFrame consisting of the
        sample rows of labels pertaining to each person's salary
        """
        # [!] TODO
        # Impute X_train data maybe?
        self.one_hot = preprocessing.OneHotEncoder(handle_unknown = 'ignore')
        features = self.one_hot.fit_transform(X_train)
        self.lg = LogisticRegression(max_iter = 1000)
        self.lg.fit(features, y_train)
        
    def classify (self, X_test):
        """
        Takes a DataFrame of rows of input attributes of census demographic
        and provides a classification for each. Note: must perform the same
        data transformations on these test rows as was done during training!
        
        :param DataFrame X_test: DataFrame of rows consisting of demographic
        attributes to be classified
        :return: A list of classifications, one for each input row X=x
        """
        # [!] TODO
        test = self.one_hot.transform(X_test)
        return self.lg.predict(test).tolist()
    
    def test_model (self, X_test, y_test):
        """
        Takes the test-set as input (2 DataFrames consisting of test demographics
        and their associated labels), classifies each, and then prints
        the classification_report on the expected vs. given labels.
        
        :param DataFrame X_test: Pandas DataFrame consisting of the
        sample rows of attributes pertaining to each individual
        :param DataFrame y_test: Pandas DataFrame consisting of the
        sample rows of labels pertaining to each person's salary
        """
        # [!] TODO
        y_actual = self.classify(X_test)
        print(metrics.classification_report(y_test, y_actual))
        
    
        
def load_and_sanitize (data_file):
    """
    Takes a path to the raw data file (a csv spreadsheet) and
    creates a new Pandas DataFrame from it with the sanitized
    data (e.g., removing leading / trailing spaces).
    NOTE: This should *not* do the preprocessing like turning continuous
    variables into discrete ones, or performing imputation -- those
    functions are handled in the SalaryPredictor constructor, and are
    used to preprocess all incoming test data as well.
    
    :param string data_file: String path to the data file csv to
    load-from and fashion a DataFrame from
    :return: The sanitized Pandas DataFrame containing the demographic
    information and labels. It is assumed that for n columns, the first
    n-1 are the inputs X and the nth column are the labels y
    """
    # [!] TODO
    # Change this list of columns to exclude for training to see possible differences among models:
    columns_to_drop = ['education_years', 'race']
    df = pd.read_csv(data_file, encoding = 'latin-1')
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.drop(columns_to_drop, axis = 1)
    df = df.apply(lambda x: x.astype(str).str.lower())
    imputer = SimpleImputer(missing_values = '?', strategy = 'most_frequent')
    for column in df:
        df[column] = imputer.fit_transform(df[column].values.reshape(-1,1))[:,0]
    return df

    
if __name__ == "__main__":
    # [!] TODO
    data = load_and_sanitize("../dat/salary.csv")
    training_data = data[data.columns[0:data.columns.size - 1]]
    class_data = data[data.columns[-1]]
    X_train, X_test, y_train, y_test = train_test_split(training_data, class_data, test_size = 0.2)
    predictor = SalaryPredictor(X_train, y_train)
    predictor.test_model(X_test, y_test)           
    
                



    
    