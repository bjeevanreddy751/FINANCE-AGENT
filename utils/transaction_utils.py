"""
Transaction Utility Module
Utility functions for transaction processing and validation.
"""

from typing import Optional, Dict, List, Any
from datetime import datetime
import re

# Date format patterns to try
DATE_FORMATS = [
    "%Y-%m-%d",      # 2025-01-15
    "%m/%d/%Y",      # 01/15/2025
    "%d/%m/%Y",      # 15/01/2025
    "%d-%m-%Y",      # 15-01-2025
    "%m-%d-%Y",      # 01-15-2025
    "%d/%m/%y",      # 15/01/25
    "%d-%m-%y",      # 15-01-25
    "%d.%m.%Y",      # 15.01.2025
    "%d.%m.%y",      # 15.01.25
    "%Y/%m/%d",      # 2025/01/15
]


def normalize_date(date_str: str) -> Optional[str]:
    """
    Normalize a date string to YYYY-MM-DD format.

    Args:
        date_str: Date string in various formats

    Returns:
        Normalized date string or None if parsing fails
    """
    if not date_str or not isinstance(date_str, str):
        return None

    date_str = date_str.strip()

    # Handle common separators and variations
    # Try each format
    for fmt in DATE_FORMATS:
        try:
            parsed = datetime.strptime(date_str, fmt)
            return parsed.strftime('%Y-%m-%d')
        except ValueError:
            continue

    # Try removing common separators and parsing
    cleaned = re.sub(r'[^0-9]', '', date_str)

    if len(cleaned) == 8:  # YYYYMMDD
        try:
            parsed = datetime.strptime(cleaned, '%Y%m%d')
            return parsed.strftime('%Y-%m-%d')
        except ValueError:
            pass

    if len(cleaned) == 6:  # YYMMDD or MMDDYY
        # Try as YYMMDD
        try:
            parsed = datetime.strptime('20' + cleaned, '%Y%m%d')
            return parsed.strftime('%Y-%m-%d')
        except ValueError:
            pass

    # Try with time component (strip time)
    if ' ' in date_str or 'T' in date_str:
        date_part = date_str.split(' ')[0].split('T')[0]
        result = normalize_date(date_part)
        if result:
            return result

    return None


def parse_amount(amount_value: Any) -> float:
    """
    Parse an amount value to float.

    Args:
        amount_value: Value to parse (string, number, etc.)

    Returns:
        Parsed float amount, or 0.0 if parsing fails
    """
    if amount_value is None:
        return 0.0

    if isinstance(amount_value, (int, float)):
        return float(amount_value)

    if isinstance(amount_value, str):
        # Remove currency symbols and whitespace
        cleaned = re.sub(r'[₹$€£,\s]', '', amount_value.strip())

        # Handle amounts in parentheses (negative)
        if cleaned.startswith('(') and cleaned.endswith(')'):
            cleaned = '-' + cleaned[1:-1]

        try:
            return float(cleaned)
        except ValueError:
            return 0.0

    try:
        return float(amount_value)
    except (ValueError, TypeError):
        return 0.0


def validate_transaction(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize a transaction dictionary.

    Args:
        transaction: Transaction dictionary

    Returns:
        Validated transaction dictionary with cleaned values
    """
    validated = {}

    # Validate date
    date_val = transaction.get('date')
    if date_val:
        normalized = normalize_date(str(date_val))
        if normalized:
            validated['date'] = normalized
    else:
        validated['date'] = None

    # Validate description
    desc_val = transaction.get('description')
    if desc_val:
        validated['description'] = str(desc_val).strip()[:200]  # Limit length
    else:
        validated['description'] = "Unknown Transaction"

    # Validate amounts
    validated['debit'] = parse_amount(transaction.get('debit'))
    validated['credit'] = parse_amount(transaction.get('credit'))

    # Ensure amount is present
    amount_val = transaction.get('amount')
    if amount_val is not None:
        validated['amount'] = parse_amount(amount_val)
    else:
        # Calculate from debit/credit
        validated['amount'] = validated['debit'] + validated['credit']

    # Validate balance
    validated['balance'] = parse_amount(transaction.get('balance'))

    return validated


def detect_invalid_transactions(transactions: List[Dict[str, Any]],
                                  min_amount: float = 0.01) -> List[Dict[str, Any]]:
    """
    Detect invalid or suspicious transactions.

    Args:
        transactions: List of transaction dictionaries
        min_amount: Minimum valid transaction amount

    Returns:
        List of invalid transaction dictionaries with reason
    """
    invalid = []

    for txn in transactions:
        issues = []

        # Check date
        if not txn.get('date'):
            issues.append("Missing date")

        # Check description
        if not txn.get('description') or len(str(txn.get('description', '')).strip()) < 2:
            issues.append("Missing/invalid description")

        # Check amount
        amount = txn.get('amount', 0)
        if amount <= min_amount:
            issues.append(f"Amount too small ({amount})")
        elif amount < 0:
            issues.append(f"Negative amount ({amount})")

        if issues:
            invalid.append({
                **txn,
                'validation_issues': issues
            })

    return invalid


def merge_transactions(existing: List[Dict[str, Any]],
                       new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge new transactions into existing list, avoiding duplicates.

    Duplicates are identified by matching date + description + amount.

    Args:
        existing: Existing transactions list
        new: New transactions to add

    Returns:
        Merged transactions list
    """
    if not existing:
        return new

    # Create signature for existing transactions
    existing_signatures = set()
    for txn in existing:
        sig = (
            txn.get('date'),
            str(txn.get('description', '')).lower().strip(),
            txn.get('amount')
        )
        existing_signatures.add(sig)

    merged = existing.copy()

    for new_txn in new:
        sig = (
            new_txn.get('date'),
            str(new_txn.get('description', '')).lower().strip(),
            new_txn.get('amount')
        )

        if sig not in existing_signatures:
            merged.append(new_txn)
            existing_signatures.add(sig)

    return merged


def format_transaction_for_display(txn: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a transaction for display in the UI.

    Args:
        txn: Transaction dictionary

    Returns:
        Formatted transaction for display
    """
    display = {
        'date': txn.get('date', 'N/A'),
        'description': txn.get('description', 'Unknown'),
        'amount': txn.get('amount', 0),
        'type': 'Credit' if (txn.get('credit') or 0) > (txn.get('debit') or 0) else 'Debit',
        'debit': txn.get('debit', 0),
        'credit': txn.get('credit', 0),
    }

    if 'balance' in txn and txn['balance'] is not None:
        display['balance'] = txn['balance']

    return display


def get_transaction_type(txn: Dict[str, Any]) -> str:
    """
    Determine transaction type (income or expense).

    Args:
        txn: Transaction dictionary

    Returns:
        'income' if credit > debit, 'expense' otherwise
    """
    credit = txn.get('credit') or 0
    debit = txn.get('debit') or 0

    return 'income' if credit > debit else 'expense'
