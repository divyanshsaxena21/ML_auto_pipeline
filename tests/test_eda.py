import unittest
import pandas as pd
from ml_autopipeline import load_data, basic_report

class TestEDA(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': ['A', 'B', 'A', 'B'],
            'target': [0, 1, 0, 1]
        })

    def test_basic_report(self):
        report = basic_report(self.df)
        self.assertEqual(report['shape'], (4, 3))
        self.assertIn('feature1', report['columns'])
        self.assertIn('target', report['columns'])
        self.assertEqual(report['missing_values']['feature1'], 0)

if __name__ == "__main__":
    unittest.main()
