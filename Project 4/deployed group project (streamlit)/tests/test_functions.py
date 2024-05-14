from functions import import_model, predict_m, load_data, load_data_eda, select_and_filter
import pickle
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch

class ImportModelTestCase(unittest.TestCase):

    @patch('pickle.load')
    @patch('builtins.open')
    def test_import_classification_model(self, mock_open, mock_pickle_load):
        mock_filename = 'pages/model_class.sav'
        mock_file = mock_open.return_value
        mock_pickle_load.return_value = 'classification_model'
        
        result = import_model('class')
        
        mock_open.assert_called_once_with(mock_filename, 'rb')
        mock_pickle_load.assert_called_once_with(mock_file)
        self.assertEqual(result, 'classification_model')

    @patch('pickle.load')
    @patch('builtins.open')
    def test_import_clustering_model(self, mock_open, mock_pickle_load):
        mock_filename = 'pages/model_clust.sav'
        mock_file = mock_open.return_value
        mock_pickle_load.return_value = 'clustering_model'
        
        result = import_model('clust')
        
        mock_open.assert_called_once_with(mock_filename, 'rb')
        mock_pickle_load.assert_called_once_with(mock_file)
        self.assertEqual(result, 'clustering_model')

    def test_invalid_model_argument(self):
        result = import_model('invalid')
        
        self.assertEqual(result, 1)
        
        
class TestLoadData(unittest.TestCase):
    def test_load_data(self):

        X_result, y_result = load_data('data/')
        
        self.assertTrue(len(X_result.columns) == 11)
        self.assertTrue(len(y_result.columns) == 1)
        
        
        
class PredictMTestCase(unittest.TestCase):
    def setUp(self):
        # Set up test data and model
        self.model = import_model('class', 'pages/')  # Replace 'YourModel' with your actual model class
        dict_data = {
            '<30': [1710, 1210],
            '30-44': [768, 768],
            '45-64': [1759, 1759],
            '65-74': [712, 712],
            '>=75': [479, 479],
            'High': [24, 24],
            'low': [9, 9],
            'Hoog': [1460, 1460],
            'Laag': [860, 860],
            'Midden': [1690, 1690],
            'QoL': [4.21, 4.21]
        }
        self.data = pd.DataFrame.from_dict(dict_data)

    def test_predict_m_low_crime(self):
        # Test prediction for low crime
        expected = 'Low crime'
        reshaped_data = np.array(self.data.iloc[1]).reshape(1, -1)
        result = predict_m(self.model, reshaped_data)
        self.assertEqual(result, expected)

    def test_predict_m_medium_crime(self):
        # Test prediction for medium crime
        expected = 'Medium crime'
        reshaped_data = np.array(self.data.iloc[0]).reshape(1, -1)
        result = predict_m(self.model, reshaped_data)
        self.assertEqual(result, expected)
        
class LoadDataEdaTestCase(unittest.TestCase):
    def test_load_eda(self):
        numbers_of_crime, population, educational_level, income_level, qol = load_data_eda('pages/data/')
        
        assert not numbers_of_crime.empty
        assert not population.empty
        assert not educational_level.empty
        assert not income_level.empty
        assert not qol.empty
        
        
class SelectAndFilterTestCase(unittest.TestCase):
    def setUp(self):
        data = {
            'Year': [2020, 2021, 2022],
            'Category': ['A', 'B', 'C'],
            'Value': [1, 2, 3]
        }   
        self.data = pd.DataFrame.from_dict(data)
    def test_select_and_filter(self):
        # Test case 1: selecting 'Category' and filtering by 'Year'
        select = 'Category'
        key = ' by Year'
        expected_output = {'Year': {0: 2020}, 'Category': {0:'A'}, 'Value': {0: 1}}
        result = select_and_filter(select, self.data, key)
        self.assertEqual(result.to_dict(), expected_output)

        # Test case 2: selecting 'Value' and filtering by 'Year'
        select = 'Value'
        key = ' by Year'
        expected_output = {'Year': {0: 2020}, 'Category': {0:'A'}, 'Value': {0: 1}}
        result = select_and_filter(select, self.data, key)
        self.assertEqual(result.to_dict(), expected_output)

        # Test case 3: selecting 'Category' and filtering without specifying 'key'
        select = 'Category'
        expected_output = {'Year': {0: 2020}, 'Category': {0:'A'}, 'Value': {0: 1}}
        result = select_and_filter(select, self.data)
        self.assertEqual(result.to_dict(), expected_output)

    def tearDown(self):
        # Clean up test data if necessary
        pass
    
if __name__ == '__main__':
    unittest.main()
