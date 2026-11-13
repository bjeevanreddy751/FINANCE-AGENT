"""
Tests for FinanceIQ Categorizer Agent
"""

import pytest
from agents.categorizer_agent import CategorizerAgent, CategorizationResult
from prompts.categorizer_prompt import get_category_from_keywords


class TestCategorizerAgent:
    """Tests for the Categorizer Agent."""

    def setup_method(self):
        """Set up test fixtures."""
        self.categorizer = CategorizerAgent()

    def test_rule_based_food_category(self):
        """Test rule-based categorization for food items."""
        transactions = [
            {"description": "Swiggy Food Order", "amount": 500, "debit": 500, "credit": 0},
            {"description": "Zomato Dinner", "amount": 800, "debit": 800, "credit": 0},
            {"description": "McDonalds Lunch", "amount": 350, "debit": 350, "credit": 0},
            {"description": "Domino's Pizza", "amount": 450, "debit": 450, "credit": 0},
        ]

        categorized = self.categorizer.apply_rule_based_categories(transactions)

        for txn in categorized:
            assert txn['category'] == "Food and Dining"

    def test_rule_based_shopping_category(self):
        """Test rule-based categorization for shopping."""
        transactions = [
            {"description": "Amazon Purchase", "amount": 2000, "debit": 2000, "credit": 0},
            {"description": "Flipkart Electronics", "amount": 5000, "debit": 5000, "credit": 0},
            {"description": "Myntra Clothing", "amount": 1500, "debit": 1500, "credit": 0},
        ]

        categorized = self.categorizer.apply_rule_based_categories(transactions)

        for txn in categorized:
            assert txn['category'] == "Shopping"

    def test_rule_based_transportation_category(self):
        """Test rule-based categorization for transportation."""
        transactions = [
            {"description": "Uber Ride", "amount": 200, "debit": 200, "credit": 0},
            {"description": "Ola Cab", "amount": 300, "debit": 300, "credit": 0},
            {"description": "Metro Commute", "amount": 100, "debit": 100, "credit": 0},
            {"description": "Petrol Fuel", "amount": 1500, "debit": 1500, "credit": 0},
        ]

        categorized = self.categorizer.apply_rule_based_categories(transactions)

        for txn in categorized:
            assert txn['category'] == "Transportation"

    def test_rule_based_entertainment_category(self):
        """Test rule-based categorization for entertainment."""
        transactions = [
            {"description": "Netflix Subscription", "amount": 799, "debit": 799, "credit": 0},
            {"description": "Spotify Premium", "amount": 149, "debit": 149, "credit": 0},
            {"description": "BookMyShow Tickets", "amount": 500, "debit": 500, "credit": 0},
        ]

        categorized = self.categorizer.apply_rule_based_categories(transactions)

        for txn in categorized:
            assert txn['category'] == "Entertainment"

    def test_rule_based_bills_category(self):
        """Test rule-based categorization for bills and utilities."""
        transactions = [
            {"description": "Electricity Bill", "amount": 1500, "debit": 1500, "credit": 0},
            {"description": "Broadband Internet", "amount": 999, "debit": 999, "credit": 0},
            {"description": "Phone Recharge", "amount": 300, "debit": 300, "credit": 0},
        ]

        categorized = self.categorizer.apply_rule_based_categories(transactions)

        for txn in categorized:
            assert txn['category'] == "Bills and Utilities"

    def test_rule_based_income_category(self):
        """Test rule-based categorization for income."""
        transactions = [
            {"description": "Salary Credit", "amount": 50000, "debit": 0, "credit": 50000},
            {"description": "Payment Received", "amount": 10000, "debit": 0, "credit": 10000},
        ]

        categorized = self.categorizer.apply_rule_based_categories(transactions)

        for txn in categorized:
            assert txn['category'] == "Salary and Income"

    def test_rule_based_transfer_category(self):
        """Test rule-based categorization for transfers."""
        transactions = [
            {"description": "Bank Transfer Sent", "amount": 5000, "debit": 5000, "credit": 0},
            {"description": "UPI Transfer", "amount": 2000, "debit": 2000, "credit": 0},
        ]

        categorized = self.categorizer.apply_rule_based_categories(transactions)

        for txn in categorized:
            assert txn['category'] == "Transfers"

    def test_individual_transaction_categorization(self):
        """Test categorization of individual transactions."""
        transaction = {
            "date": "2025-01-01",
            "description": "Amazon Shopping",
            "amount": 2000,
            "debit": 2000,
            "credit": 0
        }

        result = self.categorizer.categorize_transaction(transaction)

        assert isinstance(result, CategorizationResult)
        assert result.suggested_category == "Shopping"
        assert result.method == "rule-based"

    def test_user_correction_override(self):
        """Test that user corrections override rule-based categorization."""
        self.categorizer.set_user_correction("Old Description", "Travel")

        transaction = {
            "date": "2025-01-01",
            "description": "Old Description",
            "amount": 5000,
            "debit": 5000,
            "credit": 0
        }

        result = self.categorizer.categorize_transaction(transaction)

        assert result.suggested_category == "Travel"
        assert result.method == "user-corrected"

    def test_category_confidence_levels(self):
        """Test that confidence levels are properly assigned."""
        high_conf_txn = {"description": "Swiggy Food Order", "amount": 500, "debit": 500, "credit": 0}
        low_conf_txn = {"description": "Some Random Purchase", "amount": 100, "debit": 100, "credit": 0}

        high_result = self.categorizer.categorize_transaction(high_conf_txn)
        low_result = self.categorizer.categorize_transaction(low_conf_txn)

        # High confidence for clear matches
        assert high_result.confidence in ["high", "medium"]

        # Low confidence for unclear matches
        assert low_result.confidence in ["medium", "low"]

    def test_batch_categorization(self):
        """Test batch categorization."""
        transactions = [
            {"description": "Food Order", "amount": 500, "debit": 500, "credit": 0},
            {"description": "Shopping", "amount": 2000, "debit": 2000, "credit": 0},
            {"description": "Uber", "amount": 150, "debit": 150, "credit": 0},
        ]

        results = self.categorizer.categorize_batch(transactions)

        assert len(results) == 3
        assert all(isinstance(r, CategorizationResult) for r in results)

    def test_category_summary(self):
        """Test category summary generation."""
        transactions = [
            {"description": "Food Order", "amount": 500, "debit": 500, "credit": 0, "category": "Food and Dining"},
            {"description": "Another Food", "amount": 300, "debit": 300, "credit": 0, "category": "Food and Dining"},
            {"description": "Shopping", "amount": 2000, "debit": 2000, "credit": 0, "category": "Shopping"},
        ]

        summary = self.categorizer.get_category_summary(transactions)

        assert "Food and Dining" in summary
        assert "Shopping" in summary
        assert summary["Food and Dining"]["total"] == 800
        assert summary["Food and Dining"]["count"] == 2

    def test_ambiguous_transaction_detection(self):
        """Test detection of transactions that need review."""
        transactions = [
            {"description": "Swiggy Food", "amount": 500, "debit": 500, "credit": 0, "category": "Food and Dining", "category_confidence": "high"},
            {"description": "Unknown Purchase", "amount": 100, "debit": 100, "credit": 0, "category": "Other", "category_confidence": "low"},
        ]

        ambiguous = self.categorizer.detect_ambiguous_transactions(transactions, threshold="medium")

        assert len(ambiguous) >= 1  # At least the "Other" category one should be flagged


class TestKeywordCategorization:
    """Tests for keyword-based categorization utility."""

    def test_get_category_from_keywords_food(self):
        """Test keyword matching for food categories."""
        assert get_category_from_keywords("Swiggy order") == "Food and Dining"
        assert get_category_from_keywords("Zomato dinner") == "Food and Dining"
        assert get_category_from_keywords("McDonalds") == "Food and Dining"

    def test_get_category_from_keywords_shopping(self):
        """Test keyword matching for shopping categories."""
        assert get_category_from_keywords("Amazon purchase") == "Shopping"
        assert get_category_from_keywords("Flipkart electronics") == "Shopping"

    def test_get_category_from_keywords_transport(self):
        """Test keyword matching for transportation."""
        assert get_category_from_keywords("Uber ride") == "Transportation"
        assert get_category_from_keywords("Ola cab") == "Transportation"
        assert get_category_from_keywords("Petrol pump") == "Transportation"

    def test_get_category_from_keywords_entertainment(self):
        """Test keyword matching for entertainment."""
        assert get_category_from_keywords("Netflix subscription") == "Entertainment"
        assert get_category_from_keywords("Spotify") == "Entertainment"

    def test_get_category_from_keywords_income(self):
        """Test keyword matching for income."""
        assert get_category_from_keywords("Salary credit") == "Salary and Income"
        assert get_category_from_keywords("Payment received") == "Salary and Income"

    def test_get_category_from_keywords_unknown(self):
        """Test that unknown descriptions return 'Other'."""
        assert get_category_from_keywords("Random transaction") == "Other"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_description(self):
        """Test handling of empty descriptions."""
        categorizer = CategorizerAgent()
        result = categorizer.categorize_transaction({
            "description": "",
            "amount": 100,
            "debit": 100,
            "credit": 0
        })
        assert result.suggested_category == "Other"

    def test_very_long_description(self):
        """Test handling of very long descriptions."""
        categorizer = CategorizerAgent()
        result = categorizer.categorize_transaction({
            "description": "A" * 1000 + " Swiggy Food Order",
            "amount": 500,
            "debit": 500,
            "credit": 0
        })
        assert result.suggested_category == "Food and Dining"

    def test_special_characters_in_description(self):
        """Test handling of special characters."""
        categorizer = CategorizerAgent()
        result = categorizer.categorize_transaction({
            "description": "@@@ Swiggy @@@",
            "amount": 500,
            "debit": 500,
            "credit": 0
        })
        # Should still recognize the keyword
        assert result.suggested_category == "Food and Dining"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
