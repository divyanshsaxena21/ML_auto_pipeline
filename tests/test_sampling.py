import unittest
import pandas as pd
from ml_autopipeline import check_imbalance, apply_smote

class TestSampling(unittest.TestCase):
    def setUp(self):
        self.df_balanced = pd.DataFrame({
            'feature': list(range(10)),
            'target': [0, 1] * 5
        })
        self.df_imbalanced = pd.DataFrame({
            'feature': list(range(10)),
            'target': [0] * 9 + [1]
        })

    def test_check_imbalance_balanced(self):
        result = check_imbalance(self.df_balanced, 'target')
        self.assertFalse(result['is_imbalanced'])

    def test_check_imbalance_imbalanced(self):
        result = check_imbalance(self.df_imbalanced, 'target')
        self.assertTrue(result['is_imbalanced'])

    def test_apply_smote(self):
        X = self.df_imbalanced[['feature']]
        y = self.df_imbalanced['target']
        X_res, y_res = apply_smote(X, y)
        # After SMOTE, classes should be balanced
        self.assertEqual(sum(y_res == 0), sum(y_res == 1))

if __name__ == "__main__":
    unittest.main()
