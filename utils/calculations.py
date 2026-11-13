"""
Financial Calculations Module
Performs financial analysis and calculations on transaction data.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import numpy as np


def calculate_total_income(transactions: pd.DataFrame) -> float:
    """
    Calculate total income from transactions.

    Args:
        transactions: DataFrame of transactions

    Returns:
        Total income amount
    """
    if transactions.empty:
        return 0.0

    # Income = credits or positive amounts
    income = transactions['credit'].sum()

    # Also count positive amounts as income if no credit column distinction
    if 'amount' in transactions.columns:
        # Only count if clearly income (credit > 0)
        pass

    return float(income)


def calculate_total_expenses(transactions: pd.DataFrame) -> float:
    """
    Calculate total expenses from transactions.

    Args:
        transactions: DataFrame of transactions

    Returns:
        Total expenses amount
    """
    if transactions.empty:
        return 0.0

    # Expenses = debits or negative amounts
    expenses = transactions['debit'].sum()

    return float(expenses)


def calculate_net_savings(transactions: pd.DataFrame) -> float:
    """
    Calculate net savings (income - expenses).

    Args:
        transactions: DataFrame of transactions

    Returns:
        Net savings amount
    """
    income = calculate_total_income(transactions)
    expenses = calculate_total_expenses(transactions)
    return income - expenses


def calculate_monthly_breakdown(transactions: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate monthly income, expenses, and savings.

    Args:
        transactions: DataFrame of transactions

    Returns:
        Dictionary with monthly breakdown
    """
    if transactions.empty:
        return {}

    # Ensure date is datetime
    df = transactions.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    df = df.dropna(subset=['date'])

    if df.empty:
        return {}

    # Add month column
    df['month'] = df['date'].dt.to_period('M').astype(str)

    monthly_data = {}

    for month in sorted(df['month'].unique()):
        month_df = df[df['month'] == month]
        income = float(month_df['credit'].sum())
        expenses = float(month_df['debit'].sum())
        savings = income - expenses

        monthly_data[month] = {
            'income': income,
            'expenses': expenses,
            'savings': savings,
            'transaction_count': len(month_df)
        }

    return monthly_data


def calculate_category_breakdown(transactions: pd.DataFrame,
                                  category_col: str = 'category') -> Dict[str, Dict[str, float]]:
    """
    Calculate spending/income breakdown by category.

    Args:
        transactions: DataFrame of transactions
        category_col: Name of category column

    Returns:
        Dictionary with category-wise breakdown
    """
    if transactions.empty or category_col not in transactions.columns:
        return {}

    breakdown = defaultdict(lambda: {'expenses': 0.0, 'income': 0.0, 'count': 0})

    for _, row in transactions.iterrows():
        category = row.get(category_col, 'Uncategorized')

        if pd.isna(category) or not category:
            category = 'Uncategorized'

        amount = float(row.get('amount', 0))
        debit = float(row.get('debit', 0))
        credit = float(row.get('credit', 0))

        if credit > debit:
            breakdown[category]['income'] += credit
        else:
            breakdown[category]['expenses'] += debit

        breakdown[category]['count'] += 1

    # Convert to regular dict with proper structure
    result = {}
    for category, data in breakdown.items():
        result[category] = {
            'expenses': data['expenses'],
            'income': data['income'],
            'count': data['count'],
            'total': data['expenses'] + data['income']
        }

    # Sort by total descending
    return dict(sorted(result.items(), key=lambda x: x[1]['total'], reverse=True))


def calculate_highest_spending_category(transactions: pd.DataFrame,
                                         category_col: str = 'category') -> Optional[str]:
    """
    Find the category with highest spending.

    Args:
        transactions: DataFrame of transactions
        category_col: Name of category column

    Returns:
        Category name or None if no data
    """
    breakdown = calculate_category_breakdown(transactions, category_col)

    if not breakdown:
        return None

    # Filter to only expense categories
    expense_categories = {
        cat: data['expenses']
        for cat, data in breakdown.items()
        if data['expenses'] > 0
    }

    if not expense_categories:
        return None

    return max(expense_categories.items(), key=lambda x: x[1])[0]


def calculate_average_daily_expenditure(transactions: pd.DataFrame) -> float:
    """
    Calculate average daily expenditure.

    Args:
        transactions: DataFrame of transactions

    Returns:
        Average daily expenditure
    """
    if transactions.empty:
        return 0.0

    df = transactions.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    df = df.dropna(subset=['date', 'debit'])
    df = df[df['debit'] > 0]

    if df.empty:
        return 0.0

    # Get date range
    min_date = df['date'].min()
    max_date = df['date'].max()

    date_range = (max_date - min_date).days
    if date_range == 0:
        date_range = 1  # Avoid division by zero

    total_expenses = df['debit'].sum()
    return float(total_expenses / date_range)


def calculate_monthly_savings_rate(transactions: pd.DataFrame,
                                    month: Optional[str] = None) -> float:
    """
    Calculate savings rate for a specific month or overall.

    Savings rate = (Income - Expenses) / Income * 100

    Args:
        transactions: DataFrame of transactions
        month: Specific month in YYYY-MM format, or None for overall

    Returns:
        Savings rate as percentage
    """
    if transactions.empty:
        return 0.0

    df = transactions.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    df = df.dropna(subset=['date'])

    if month:
        # Filter to specific month
        start_date = f"{month}-01"
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            if month.endswith('-12'):
                end = datetime.strptime(f"{int(month[:4])+1}-01-01", '%Y-%m-%d')
            else:
                end = datetime.strptime(f"{month}-01", '%Y-%m-%d') + timedelta(days=32)
                end = end.replace(day=1)
            df = df[(df['date'] >= start) & (df['date'] < end)]
        except ValueError:
            pass

    if df.empty:
        return 0.0

    income = float(df['credit'].sum())
    expenses = float(df['debit'].sum())

    if income <= 0:
        return 0.0

    return ((income - expenses) / income) * 100


def calculate_month_over_month_change(transactions: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Calculate month-over-month spending changes.

    Args:
        transactions: DataFrame of transactions

    Returns:
        Dictionary with month-over-month changes
    """
    monthly_data = calculate_monthly_breakdown(transactions)
    months = sorted(monthly_data.keys())

    if len(months) < 2:
        return {}

    changes = {}

    for i in range(1, len(months)):
        prev_month = months[i - 1]
        curr_month = months[i]

        prev_expenses = monthly_data[prev_month]['expenses']
        curr_expenses = monthly_data[curr_month]['expenses']

        if prev_expenses > 0:
            percent_change = ((curr_expenses - prev_expenses) / prev_expenses) * 100
        else:
            percent_change = 0.0

        changes[curr_month] = {
            'previous_month': prev_month,
            'previous_expenses': prev_expenses,
            'current_expenses': curr_expenses,
            'absolute_change': curr_expenses - prev_expenses,
            'percent_change': percent_change,
            'trend': 'increase' if percent_change > 0 else 'decrease'
        }

    return changes


def detect_unusual_transactions(transactions: pd.DataFrame,
                                 std_dev_multiplier: float = 2.0) -> List[Dict[str, Any]]:
    """
    Detect unusual transactions based on statistical analysis.

    Args:
        transactions: DataFrame of transactions
        std_dev_multiplier: Number of standard deviations for threshold

    Returns:
        List of unusual transactions with explanation
    """
    if transactions.empty:
        return []

    df = transactions.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    df = df.dropna(subset=['debit'])
    df = df[df['debit'] > 0]

    if df.empty or len(df) < 3:
        return []

    mean_amount = df['debit'].mean()
    std_amount = df['debit'].std()

    if pd.isna(std_amount) or std_amount == 0:
        return []

    threshold = mean_amount + (std_dev_multiplier * std_amount)

    unusual = df[df['debit'] > threshold]

    results = []
    for _, row in unusual.iterrows():
        results.append({
            'date': row.get('date').strftime('%Y-%m-%d') if pd.notna(row.get('date')) else None,
            'description': row.get('description', 'Unknown'),
            'amount': float(row['debit']),
            'reason': f"Amount ({row['debit']:.2f}) is {(row['debit'] / mean_amount - 1) * 100:.1f}% above average",
            'avg_amount': float(mean_amount),
            'threshold': float(threshold)
        })

    return results


def calculate_running_balance(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate running balance from transactions.

    Args:
        transactions: DataFrame of transactions (must have date column)

    Returns:
        DataFrame with running balance column added
    """
    if transactions.empty:
        return transactions

    df = transactions.copy()

    # Convert date to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)

    # Calculate running balance
    df['running_balance'] = 0.0

    balance = 0.0
    for idx, row in df.iterrows():
        credit = float(row.get('credit', 0) or 0)
        debit = float(row.get('debit', 0) or 0)
        balance += credit - debit
        df.at[idx, 'running_balance'] = balance

    return df


def get_income_vs_expenses_by_month(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Get monthly income vs expenses data for charting.

    Args:
        transactions: DataFrame of transactions

    Returns:
        DataFrame with month, income, expenses columns
    """
    monthly = calculate_monthly_breakdown(transactions)

    if not monthly:
        return pd.DataFrame(columns=['month', 'income', 'expenses', 'savings'])

    data = []
    for month, values in sorted(monthly.items()):
        data.append({
            'month': month,
            'income': values['income'],
            'expenses': values['expenses'],
            'savings': values['savings']
        })

    return pd.DataFrame(data)


def filter_transactions(transactions: pd.DataFrame,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        category: Optional[str] = None,
                        min_amount: Optional[float] = None,
                        max_amount: Optional[float] = None,
                        search_query: Optional[str] = None) -> pd.DataFrame:
    """
    Filter transactions based on various criteria.

    Args:
        transactions: DataFrame of transactions
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        category: Category to filter by
        min_amount: Minimum transaction amount
        max_amount: Maximum transaction amount
        search_query: Text to search in description

    Returns:
        Filtered DataFrame
    """
    df = transactions.copy()

    if df.empty:
        return df

    # Convert date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Filter by date range
    if start_date:
        start = pd.to_datetime(start_date)
        df = df[df['date'] >= start]

    if end_date:
        end = pd.to_datetime(end_date)
        df = df[df['date'] <= end]

    # Filter by category
    if category and category != 'All':
        df = df[df.get('category', '') == category]

    # Filter by amount range
    if min_amount is not None:
        df = df[df['amount'] >= min_amount]

    if max_amount is not None:
        df = df[df['amount'] <= max_amount]

    # Filter by search query
    if search_query:
        search = search_query.lower()
        df = df[df['description'].str.lower().str.contains(search, na=False)]

    return df.reset_index(drop=True)


def get_recent_transactions(transactions: pd.DataFrame,
                            limit: int = 10) -> pd.DataFrame:
    """
    Get most recent transactions.

    Args:
        transactions: DataFrame of transactions
        limit: Number of transactions to return

    Returns:
        DataFrame with recent transactions
    """
    if transactions.empty:
        return transactions

    df = transactions.copy()

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    df = df.sort_values('date', ascending=False)
    return df.head(limit).reset_index(drop=True)
