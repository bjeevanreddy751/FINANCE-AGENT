"""
Advisor Agent Prompts
Prompt templates for financial advice and insights generation.
"""

from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# System prompt for financial advisor
ADVISOR_SYSTEM_PROMPT = """You are a FinanceIQ Financial Advisor Agent.
Your task is to provide personalized, evidence-based financial insights and recommendations.

KEY PRINCIPLES:
1. Base ALL insights on the user's actual uploaded transaction data
2. Never invent financial figures - use exact numbers from the data
3. Be specific: "You spent ₹X on Food" not "You spent a lot on food"
4. Be helpful but not preachy
5. Always include specific transaction references when making claims
6. Never present suggestions as guaranteed financial advice
7. Focus on actionable, achievable recommendations

CATEGORIES YOU CAN REFERENCE:
- Food and Dining
- Shopping
- Transportation
- Groceries
- Entertainment
- Healthcare
- Education
- Bills and Utilities
- Rent
- Travel
- Salary and Income
- Transfers
- Other

When analyzing data:
- Compare months to identify trends
- Identify overspending patterns
- Highlight unusual transactions
- Calculate actual savings rates
- Provide specific, actionable suggestions

IMPORTANT: If the user asks about specific transactions or amounts, you must:
1. Check the provided data context
2. Reference exact figures from the data
3. If the data isn't available, say so clearly
"""

# Human prompt for generating insights
ADVISOR_HUMAN_INSIGHTS_PROMPT = """Based on the financial data below, generate personalized insights:

USER'S FINANCIAL DATA:
{financial_data}

SPECIFIC METRICS:
- Total Income: {total_income}
- Total Expenses: {total_expenses}
- Net Savings: {net_savings}
- Savings Rate: {savings_rate}%
- Highest Spending Category: {highest_category} ({highest_category_amount})
- Average Daily Spending: {avg_daily}

CATEGORY BREAKDOWN:
{category_breakdown}

MONTHLY TRENDS:
{monthly_trends}

UNUSUAL TRANSACTIONS:
{unusual_transactions}

Please provide:
1. 3-5 key insights about the user's spending patterns
2. Specific areas where expenses could be optimized
3. At least 2 actionable savings recommendations
4. A brief positive observation about their financial habits

Format your response clearly with bullet points and specific numbers.
"""

# Prompt for answering specific questions
ADVISOR_QUESTION_PROMPT = """You are a FinanceIQ Financial Assistant.

USER'S FINANCIAL CONTEXT:
{context}

Answer the following question based ONLY on the data above:
{query}

If the question requires information not available in the context, clearly state that
the information isn't available in the uploaded statement.

Provide specific, data-backed answers.
"""


# Create the insight generation chain
advisor_insights_prompt = ChatPromptTemplate.from_messages([
    ("system", ADVISOR_SYSTEM_PROMPT),
    ("human", ADVISOR_HUMAN_INSIGHTS_PROMPT),
])

# Create the Q&A prompt chain
advisor_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", ADVISOR_SYSTEM_PROMPT),
    ("human", ADVISOR_QUESTION_PROMPT),
])

# Intent-specific prompts
INTENT_SPending_ANALYSIS_PROMPT = """Analyze the user's spending patterns based on this data:

{context}

Focus on:
1. Overall spending trends
2. Category-wise spending breakdown
3. Spending vs income comparison
4. Key spending patterns

Provide a clear spending analysis with specific numbers."""

INTENT_TRANSACTION_SEARCH_PROMPT = """Based on the transaction data below, help find transactions matching the user's request:

{context}

User wants to find: {query}

Search the transactions and provide relevant results with exact figures."""

INTENT_MONTHLY_COMPARISON_PROMPT = """Compare the user's financial performance between months:

{context}

Compare: {comparison_query}

Provide:
- Month-by-month breakdown
- Income comparison
- Expense comparison
- Savings comparison
- Key differences and trends"""


def get_insights_prompt(financial_data: str, metrics: Dict,
                         category_breakdown: str, monthly_trends: str,
                         unusual_transactions: str) -> ChatPromptTemplate:
    """Generate a comprehensive insights prompt."""
    return advisor_insights_prompt.format(
        financial_data=financial_data,
        total_income=metrics.get('total_income', 0),
        total_expenses=metrics.get('total_expenses', 0),
        net_savings=metrics.get('net_savings', 0),
        savings_rate=metrics.get('savings_rate', 0),
        highest_category=metrics.get('highest_category', 'N/A'),
        highest_category_amount=metrics.get('highest_category_amount', 0),
        avg_daily=metrics.get('avg_daily', 0),
        category_breakdown=category_breakdown,
        monthly_trends=monthly_trends,
        unusual_transactions=unusual_transactions
    )


def get_question_prompt(context: str, query: str) -> ChatPromptTemplate:
    """Generate a Q&A prompt."""
    return advisor_qa_prompt.format(context=context, query=query)


def format_financial_context(transactions_df, analysis_results: Dict) -> str:
    """Format transaction data and analysis into context for the LLM."""
    context_parts = []

    # Basic stats
    context_parts.append("## FINANCIAL SUMMARY")
    context_parts.append(f"Total Income: ₹{analysis_results.get('total_income', 0):,.2f}")
    context_parts.append(f"Total Expenses: ₹{analysis_results.get('total_expenses', 0):,.2f}")
    context_parts.append(f"Net Savings: ₹{analysis_results.get('net_savings', 0):,.2f}")
    context_parts.append("")

    # Monthly breakdown
    if analysis_results.get('monthly_breakdown'):
        context_parts.append("## MONTHLY BREAKDOWN")
        for month, data in analysis_results['monthly_breakdown'].items():
            context_parts.append(f"\n### {month}")
            context_parts.append(f"  Income: ₹{data['income']:,.2f}")
            context_parts.append(f"  Expenses: ₹{data['expenses']:,.2f}")
            context_parts.append(f"  Savings: ₹{data['savings']:,.2f}")
        context_parts.append("")

    # Category breakdown
    if analysis_results.get('category_breakdown'):
        context_parts.append("## CATEGORY BREAKDOWN")
        for category, data in list(analysis_results['category_breakdown'].items())[:10]:
            context_parts.append(f"  - {category}: ₹{data['total']:,.2f} ({data['count']} transactions)")
        context_parts.append("")

    # Recent transactions
    if transactions_df is not None and len(transactions_df) > 0:
        context_parts.append("## RECENT TRANSACTIONS (Last 10)")
        recent = transactions_df.head(10)
        for _, row in recent.iterrows():
            context_parts.append(
                f"  - {row.get('date', 'N/A')}: {row.get('description', 'N/A')} - ₹{row.get('amount', 0):,.2f}"
            )
        context_parts.append("")

    return "\n".join(context_parts)


def format_category_breakdown_for_prompt(category_data: Dict) -> str:
    """Format category breakdown for inclusion in prompts."""
    lines = []
    for category, data in sorted(category_data.items(), key=lambda x: x[1]['total'], reverse=True):
        lines.append(f"- {category}: ₹{data['total']:,.2f} ({data['count']} txns)")
    return "\n".join(lines)


def format_unusual_transactions_for_prompt(unusual_txns: List[Dict]) -> str:
    """Format unusual transactions for prompts."""
    if not unusual_txns:
        return "No unusual transactions detected."

    lines = ["Unusual transactions detected:"]
    for txn in unusual_txns[:5]:  # Limit to 5
        lines.append(f"- {txn['date']}: {txn['description']} - ₹{txn['amount']:,.2f}")
        lines.append(f"  Reason: {txn['reason']}")

    return "\n".join(lines)
