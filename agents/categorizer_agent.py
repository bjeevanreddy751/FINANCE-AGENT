"""
Transaction Categorization Agent
Categorizes transactions using rule-based and LLM-based classification.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pandas as pd

from prompts.categorizer_prompt import (
    get_categorizer_prompt,
    get_category_from_keywords,
    RULE_BASED_KEYWORDS,
    CATEGORIES
)
from utils.transaction_utils import get_transaction_type


@dataclass
class CategorizationResult:
    """Result of categorizing a single transaction."""
    transaction_id: str
    description: str
    suggested_category: str
    confidence: str  # 'high', 'medium', 'low'
    method: str  # 'rule-based', 'llm-based', 'user-corrected'
    reasoning: str = ""


class CategorizerAgent:
    """
    Agent for categorizing financial transactions.

    Uses a hybrid approach:
    1. Rule-based keyword matching (fast, reliable for common patterns)
    2. LLM-based classification (for edge cases and better accuracy)
    3. User corrections (stored and respected)
    """

    def __init__(self):
        self.user_corrections: Dict[str, str] = {}  # description -> category mapping
        self.transactions_df: pd.DataFrame = None

    def categorize_transaction(self, transaction: Dict,
                               use_llm: bool = False,
                               llm=None) -> CategorizationResult:
        """
        Categorize a single transaction.

        Args:
            transaction: Transaction dictionary
            use_llm: Whether to use LLM for classification
            llm: Optional LLM instance

        Returns:
            CategorizationResult with category and confidence
        """
        txn_id = transaction.get('id') or transaction.get('date', '')
        description = transaction.get('description', '')

        # Check user corrections first
        if description in self.user_corrections:
            return CategorizationResult(
                transaction_id=txn_id,
                description=description,
                suggested_category=self.user_corrections[description],
                confidence='high',
                method='user-corrected',
                reasoning='User-corrected category'
            )

        # Try rule-based classification
        rule_category = get_category_from_keywords(description)

        # If it's income (credit > debit), special handling
        credit = transaction.get('credit', 0) or 0
        debit = transaction.get('debit', 0) or 0

        if credit > debit:
            # Income transaction
            if 'salary' in description.lower() or 'credit' in description.lower():
                return CategorizationResult(
                    transaction_id=txn_id,
                    description=description,
                    suggested_category='Salary and Income',
                    confidence='high',
                    method='rule-based',
                    reasoning='Salary/Income credit detected'
                )
            elif 'transfer' in description.lower():
                return CategorizationResult(
                    transaction_id=txn_id,
                    description=description,
                    suggested_category='Transfers',
                    confidence='high',
                    method='rule-based',
                    reasoning='Transfer detected'
                )
            else:
                return CategorizationResult(
                    transaction_id=txn_id,
                    description=description,
                    suggested_category='Salary and Income',
                    confidence='medium',
                    method='rule-based',
                    reasoning='Credit transaction categorized as income'
                )

        # Check if rule-based confidence is high
        if self._is_rule_confident(description, rule_category):
            return CategorizationResult(
                transaction_id=txn_id,
                description=description,
                suggested_category=rule_category,
                confidence='high',
                method='rule-based',
                reasoning=f"Matched keywords for {rule_category}"
            )

        # Use LLM if available and requested
        if use_llm and llm is not None:
            llm_result = self._categorize_with_llm(transaction, llm)
            if llm_result:
                return llm_result

        # Fall back to rule-based with medium confidence
        return CategorizationResult(
            transaction_id=txn_id,
            description=description,
            suggested_category=rule_category if rule_category != 'Other' else 'Other',
            confidence='medium',
            method='rule-based',
            reasoning=f"No strong keyword match, defaulting to {rule_category}"
        )

    def _is_rule_confident(self, description: str, category: str) -> bool:
        """Check if rule-based classification is confident."""
        if category == 'Other':
            return False

        description_lower = description.lower()

        # Check for strong keyword matches
        keywords = RULE_BASED_KEYWORDS.get(category, [])
        matches = sum(1 for kw in keywords if kw in description_lower)

        # Confidence if multiple keywords match
        return matches >= 2

    def _categorize_with_llm(self, transaction: Dict,
                              llm) -> Optional[CategorizationResult]:
        """
        Categorize using LLM.

        Args:
            transaction: Transaction dictionary
            llm: LangChain LLM instance

        Returns:
            CategorizationResult or None if LLM fails
        """
        try:
            # Create prompt
            prompt = get_categorizer_prompt(
                date=transaction.get('date', ''),
                description=transaction.get('description', ''),
                amount=transaction.get('amount', 0),
                current_category='Uncategorized'
            )

            # Get response
            response = llm.invoke(prompt)

            # Parse response
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Try to extract category from response
            for category in CATEGORIES:
                if category.lower() in response_text.lower():
                    return CategorizationResult(
                        transaction_id=transaction.get('id', ''),
                        description=transaction.get('description', ''),
                        suggested_category=category,
                        confidence='medium',
                        method='llm-based',
                        reasoning='LLM classified based on transaction context'
                    )

            return None

        except Exception as e:
            print(f"LLM categorization error: {e}")
            return None

    def categorize_batch(self, transactions: List[Dict],
                         use_llm: bool = False,
                         llm=None) -> List[CategorizationResult]:
        """
        Categorize multiple transactions.

        Args:
            transactions: List of transaction dictionaries
            use_llm: Whether to use LLM
            llm: Optional LLM instance

        Returns:
            List of CategorizationResult objects
        """
        results = []

        for i, txn in enumerate(transactions):
            txn_with_id = {**txn, 'id': f"txn_{i}"}
            result = self.categorize_transaction(txn_with_id, use_llm, llm)
            results.append(result)

        return results

    def apply_categories_to_df(self, df: pd.DataFrame,
                                results: List[CategorizationResult]) -> pd.DataFrame:
        """
        Apply categorization results to a DataFrame.

        Args:
            df: Transaction DataFrame
            results: List of CategorizationResult objects

        Returns:
            DataFrame with category column added
        """
        df = df.copy()
        df['category'] = 'Uncategorized'
        df['category_source'] = 'unclassified'

        for result in results:
            # Find matching row
            for idx, row in df.iterrows():
                if (row.get('date') == result.transaction_id or
                    row.get('description') == result.description):
                    df.at[idx, 'category'] = result.suggested_category
                    df.at[idx, 'category_source'] = result.method
                    break

        return df

    def apply_rule_based_categories(self, transactions: List[Dict]) -> List[Dict]:
        """
        Apply rule-based categories to transactions.

        Args:
            transactions: List of transaction dictionaries

        Returns:
            Transactions with categories added
        """
        categorized = []

        for i, txn in enumerate(transactions):
            txn_copy = txn.copy()
            result = self.categorize_transaction({**txn, 'id': f"txn_{i}"})

            txn_copy['category'] = result.suggested_category
            txn_copy['category_source'] = result.method
            txn_copy['category_confidence'] = result.confidence

            categorized.append(txn_copy)

        return categorized

    def set_user_correction(self, description: str, category: str):
        """
        Set a user correction for a transaction description.

        Args:
            description: Transaction description (or part of it)
            category: Correct category
        """
        self.user_corrections[description] = category

    def get_user_corrections(self) -> Dict[str, str]:
        """Get all user corrections."""
        return self.user_corrections.copy()

    def clear_user_corrections(self):
        """Clear all user corrections."""
        self.user_corrections = {}

    def get_category_summary(self, transactions: List[Dict]) -> Dict[str, Dict]:
        """
        Get summary of transactions by category.

        Args:
            transactions: List of categorized transactions

        Returns:
            Category-wise summary
        """
        summary = {}

        for txn in transactions:
            category = txn.get('category', 'Uncategorized')
            amount = txn.get('amount', 0)
            credit = txn.get('credit', 0) or 0
            debit = txn.get('debit', 0) or 0

            if category not in summary:
                summary[category] = {
                    'total': 0,
                    'expenses': 0,
                    'income': 0,
                    'count': 0
                }

            summary[category]['total'] += amount
            summary[category]['count'] += 1

            if credit > debit:
                summary[category]['income'] += credit
            else:
                summary[category]['expenses'] += debit

        return summary

    def detect_ambiguous_transactions(self, transactions: List[Dict],
                                       threshold: str = 'medium') -> List[Dict]:
        """
        Find transactions that might need user review.

        Args:
            transactions: List of transactions
            threshold: Minimum confidence to consider ('low', 'medium', 'high')

        Returns:
            List of potentially miscategorized transactions
        """
        confidence_levels = {'high': 3, 'medium': 2, 'low': 1}
        min_confidence = confidence_levels.get(threshold, 2)

        ambiguous = []

        for txn in transactions:
            confidence = txn.get('category_confidence', 'low')
            conf_value = confidence_levels.get(confidence, 1)

            if conf_value < min_confidence or txn.get('category') == 'Other':
                ambiguous.append({
                    **txn,
                    'review_reason': f"Low confidence ({confidence}) categorization"
                })

        return ambiguous


# For LCEL integration
from langchain_core.tools import tool

@tool
def categorize_transactions_tool(transactions_json: str) -> str:
    """
    Categorize a batch of transactions.

    Args:
        transactions_json: JSON array of transactions

    Returns:
        JSON array with categories added
    """
    import json
    transactions = json.loads(transactions_json)
    agent = CategorizerAgent()
    categorized = agent.apply_rule_based_categories(transactions)
    return json.dumps(categorized, indent=2, default=str)
