import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from finance_dsl import FAIL, INVALID, PASS, parse_expression, validate, validate_expression


class FinanceDSLTests(unittest.TestCase):
    def test_simple_interest(self):
        result = validate(
            {
                "formula": "simple_interest",
                "inputs": {"principal": 10000, "rate": "5%", "periods": 2},
                "claimed": 1000,
                "unit": "CNY",
            }
        )
        self.assertEqual(result.status, PASS)
        self.assertEqual(result.expected, "1000.00")

    def test_simple_interest_future_value(self):
        result = validate_expression(
            "simple_interest_future_value(principal=10000, rate=5%, periods=2) = 11000"
        )
        self.assertEqual(result.status, PASS)

    def test_compound_interest(self):
        result = validate_expression(
            "compound_interest(principal=10000, rate=5%, periods=2) = 11025"
        )
        self.assertEqual(result.status, PASS)
        self.assertEqual(result.expected, "11025.0000")

    def test_compound_interest_rejects_wrong_claim(self):
        result = validate_expression(
            "compound_interest(principal=10000, rate=5%, periods=2) = 11000"
        )
        self.assertEqual(result.status, FAIL)
        self.assertEqual(result.absolute_error, "25.0000")

    def test_holding_period_return(self):
        result = validate(
            {
                "formula": "holding_period_return",
                "inputs": {
                    "beginning_value": 100,
                    "ending_value": 108,
                    "income": 2,
                },
                "claimed": "10%",
            }
        )
        self.assertEqual(result.status, PASS)
        self.assertEqual(result.expected, "0.1")

    def test_accounting_equation(self):
        result = validate(
            {
                "formula": "accounting_equation",
                "inputs": {"assets": 1000, "liabilities": 600, "equity": 400},
            }
        )
        self.assertEqual(result.status, PASS)
        self.assertEqual(result.claimed, "1000")

    def test_accounting_equation_failure(self):
        result = validate(
            {
                "formula": "accounting_equation",
                "inputs": {"assets": 900, "liabilities": 600, "equity": 400},
            }
        )
        self.assertEqual(result.status, FAIL)

    def test_ambiguous_rate_is_rejected(self):
        result = validate(
            {
                "formula": "simple_interest",
                "inputs": {"principal": 1000, "rate": 5, "periods": 1},
                "claimed": 50,
            }
        )
        self.assertEqual(result.status, INVALID)
        self.assertIn("ambiguous", result.error)

    def test_missing_field_is_invalid(self):
        result = validate(
            {
                "formula": "compound_interest",
                "inputs": {"principal": 1000, "rate": "5%"},
                "claimed": 1050,
            }
        )
        self.assertEqual(result.status, INVALID)
        self.assertIn("periods", result.error)

    def test_expression_parser_rejects_python(self):
        result = validate_expression(
            "compound_interest(principal=__import__('os'), rate=5%, periods=2) = 1"
        )
        self.assertEqual(result.status, INVALID)

    def test_parse_expression(self):
        request = parse_expression(
            "holding_period_return(beginning_value=100, ending_value=108, income=2) = 10%"
        )
        self.assertEqual(request["formula"], "holding_period_return")
        self.assertEqual(request["inputs"]["income"], "2")
        self.assertEqual(request["claimed"], "10%")


if __name__ == "__main__":
    unittest.main()
