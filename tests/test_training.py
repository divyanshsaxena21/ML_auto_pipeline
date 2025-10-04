import unittest
import pandas as pd
from ml_autopipeline import train_models

class TestTraining(unittest.TestCase):
    def setUp(self):
        # Create balanced dataset
        self.df = pd.DataFrame({
            'feat1': [1, 2, 3, 4, 5, 6],
            'feat2': [5, 4, 3, 2, 1, 0],
            'target': [0, 1, 0, 1, 0, 1]
        })
        self.X = self.df[['feat1', 'feat2']]
        self.y = self.df['target']

    def test_train_models_returns_results(self):
        results = train_models(self.X, self.y)
        self.assertIn('Logistic Regression', results)
        self.assertIn('Accuracy', results['Logistic Regression'])

if __name__ == "__main__":
    unittest.main()
