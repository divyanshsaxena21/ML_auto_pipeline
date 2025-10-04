import unittest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TestEvaluationMetrics(unittest.TestCase):

    def setUp(self):
        # Example ground truth and predictions
        self.y_true = [0, 1, 0, 1, 1, 0]
        self.y_pred_good = [0, 1, 0, 1, 0, 0]  # Mostly correct
        self.y_pred_bad = [1, 0, 1, 0, 0, 1]   # Mostly wrong

    def test_accuracy(self):
        acc_good = accuracy_score(self.y_true, self.y_pred_good)
        acc_bad = accuracy_score(self.y_true, self.y_pred_bad)
        self.assertGreater(acc_good, acc_bad)

    def test_precision(self):
        prec_good = precision_score(self.y_true, self.y_pred_good, average='weighted', zero_division=0)
        prec_bad = precision_score(self.y_true, self.y_pred_bad, average='weighted', zero_division=0)
        self.assertGreater(prec_good, prec_bad)

    def test_recall(self):
        rec_good = recall_score(self.y_true, self.y_pred_good, average='weighted', zero_division=0)
        rec_bad = recall_score(self.y_true, self.y_pred_bad, average='weighted', zero_division=0)
        self.assertGreater(rec_good, rec_bad)

    def test_f1(self):
        f1_good = f1_score(self.y_true, self.y_pred_good, average='weighted', zero_division=0)
        f1_bad = f1_score(self.y_true, self.y_pred_bad, average='weighted', zero_division=0)
        self.assertGreater(f1_good, f1_bad)

if __name__ == "__main__":
    unittest.main()
