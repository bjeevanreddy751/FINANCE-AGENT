"""
Financial Advisor Agent
Generates personalized, evidence-based financial insights and recommendations.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

# LangChain imports
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agents.analysis_agent import AnalysisAgent, AnalysisResult
from prompts.advisor_prompt import (
    advisor_insights_prompt,
    advisor_qa_prompt,
    get_insights_prompt,
    get_question_prompt,
    format_financial_context,
    format_category_breakdown_for_prompt,
    format_unusual_transactions_for_prompt
)
from prompts.intent_prompt import (
    detect_intent_keyword_based,
    route_to_agent
)
from rag.rag_pipeline import FinancialRAGPipeline, get_rag_pipeline


@dataclass
class AdvisorResponse:
    """Response from the financial advisor."""
    answer: str
    sources: List[Dict]
    intent: str
    confidence: str
    suggestions: List[str]
    warnings: List[str]


class AdvisorAgent:
    """
    Agent for generating personalized financial advice.

    Key principles:
    - All insights based on actual uploaded data
    - Never invents financial figures
    - Specific, actionable recommendations
    - Evidence-backed claims with transaction references
    """

    def __init__(self, llm: BaseChatModel = None):
        self.llm = llm
        self.analysis_agent = AnalysisAgent()
        self.rag_pipeline = get_rag_pipeline()
        self.rag_pipeline.set_llm(llm) if llm else None

    def set_llm(self, llm: BaseChatModel):
        """Set the LLM for the advisor."""
        self.llm = llm
        self.rag_pipeline.set_llm(llm)

    def set_analysis(self, analysis: AnalysisResult):
        """Set the analysis results to base advice on."""
        self.analysis_agent.last_analysis = analysis

    def get_insights(self, analysis: AnalysisResult) -> AdvisorResponse:
        """
        Generate comprehensive financial insights.

        Args:
            analysis: Analysis results to base insights on

        Returns:
            AdvisorResponse with insights and suggestions
        """
        if not analysis or analysis.transaction_count == 0:
            return AdvisorResponse(
                answer="No financial data available for analysis. Please upload your bank statements.",
                sources=[],
                intent="general",
                confidence="low",
                suggestions=[],
                warnings=["No transaction data to analyze"]
            )

        self.analysis_agent.last_analysis = analysis

        # Build financial context
        financial_data = format_financial_context(
            None,  # Would pass DataFrame if available
            {
                'total_income': analysis.total_income,
                'total_expenses': analysis.total_expenses,
                'net_savings': analysis.net_savings,
                'savings_rate': analysis.savings_rate,
                'highest_category': analysis.highest_spending_category or 'N/A',
                'highest_category_amount': analysis.highest_category_amount,
                'avg_daily': analysis.average_daily_expenditure,
                'monthly_breakdown': analysis.monthly_breakdown,
                'category_breakdown': analysis.category_breakdown,
            }
        )

        # Format category breakdown
        category_breakdown_str = format_category_breakdown_for_prompt(
            analysis.category_breakdown
        )

        # Format monthly trends
        monthly_trends = self._format_monthly_trends(analysis.monthly_breakdown)

        # Format unusual transactions
        unusual_str = format_unusual_transactions_for_prompt(
            analysis.unusual_transactions
        )

        # Generate insights using LLM if available
        if self.llm:
            try:
                prompt = get_insights_prompt(
                    financial_data=financial_data,
                    metrics={
                        'total_income': analysis.total_income,
                        'total_expenses': analysis.total_expenses,
                        'net_savings': analysis.net_savings,
                        'savings_rate': analysis.savings_rate,
                        'highest_category': analysis.highest_spending_category or 'N/A',
                        'highest_category_amount': analysis.highest_category_amount,
                        'avg_daily': analysis.average_daily_expenditure,
                    },
                    category_breakdown=category_breakdown_str,
                    monthly_trends=monthly_trends,
                    unusual_transactions=unusual_str
                )

                response = self.llm.invoke(prompt)
                answer = response.content if hasattr(response, 'content') else str(response)

                return AdvisorResponse(
                    answer=answer,
                    sources=self._extract_sources(analysis),
                    intent="insight_generation",
                    confidence="high",
                    suggestions=self._generate_suggestions(analysis),
                    warnings=self._generate_warnings(analysis)
                )

            except Exception as e:
                print(f"LLM insight generation error: {e}")

        # Fall back to rule-based insights
        return self._generate_rule_based_insights(analysis)

    def answer_question(self, question: str,
                         analysis: AnalysisResult = None) -> AdvisorResponse:
        """
        Answer a specific financial question.

        Args:
            question: User's question
            analysis: Optional analysis results

        Returns:
            AdvisorResponse with answer
        """
        # Detect intent
        intent_result = detect_intent_keyword_based(question)

        # Get analysis if not provided
        if analysis is None:
            analysis = self.analysis_agent.last_analysis

        # Check if we have data
        if not analysis or analysis.transaction_count == 0:
            return AdvisorResponse(
                answer="I don't have any financial data to answer your question. Please upload your bank statements first.",
                sources=[],
                intent=intent_result['intent'],
                confidence="low",
                suggestions=["Upload bank statements to get answers"],
                warnings=["No data available"]
            )

        # Answer based on intent
        answer = self._answer_by_intent(question, intent_result['intent'], analysis)

        return AdvisorResponse(
            answer=answer,
            sources=self._extract_sources(analysis),
            intent=intent_result['intent'],
            confidence=intent_result['confidence'],
            suggestions=self._generate_suggestions(analysis),
            warnings=[]
        )

    def _answer_by_intent(self, question: str, intent: str,
                           analysis: AnalysisResult) -> str:
        """Answer question based on detected intent."""
        q = question.lower()

        if intent == "spending_analysis" or any(w in q for w in ['spend', 'expense', 'category']):
            return self._answer_spending_question(question, analysis)

        elif intent == "transaction_search" or any(w in q for w in ['find', 'search', 'show', 'which', 'transaction']):
            return self._answer_transaction_search(question, analysis)

        elif intent == "monthly_comparison" or any(w in q for w in ['compare', 'month', 'versus', 'vs']):
            return self._answer_monthly_comparison(question, analysis)

        elif intent == "budget_recommendations" or any(w in q for w in ['save', 'budget', 'recommend', 'reduce', 'saving']):
            return self._answer_budget_question(question, analysis)

        elif intent == "income_analysis" or any(w in q for w in ['income', 'salary', 'earn']):
            return self._answer_income_question(question, analysis)

        else:
            return self._answer_general_question(question, analysis)

    def _answer_spending_question(self, question: str,
                                   analysis: AnalysisResult) -> str:
        """Answer spending-related questions."""
        total_expenses = analysis.total_expenses
        highest_cat = analysis.highest_spending_category
        highest_amount = analysis.highest_category_amount

        if any(w in question.lower() for w in ['total', 'how much', 'overall']):
            return (
                f"Based on your uploaded statements, your **total expenses** are "
                f"**₹{total_expenses:,.2f}** across {analysis.transaction_count} transactions."
            )

        elif any(w in question.lower() for w in ['highest', 'most', 'category']):
            if highest_cat:
                return (
                    f"Your **highest spending category** is **{highest_cat}** with "
                    f"**₹{highest_amount:,.2f}** in expenses. "
                    f"This represents {(highest_amount / total_expenses * 100):.1f}% of your total spending."
                )
            return "I don't have categorized data to identify your highest spending category."

        elif any(w in question.lower() for w in ['average', 'daily', 'per day']):
            return (
                f"Your **average daily expenditure** is **₹{analysis.average_daily_expenditure:,.2f}** "
                f"based on your transaction history."
            )

        elif any(w in question.lower() for w in ['category', 'breakdown']):
            if analysis.category_breakdown:
                breakdown = analysis.category_breakdown
                lines = ["**Your Spending by Category:**\n"]
                for cat, data in sorted(breakdown.items(), key=lambda x: x[1].get('expenses', 0), reverse=True)[:5]:
                    if data.get('expenses', 0) > 0:
                        pct = (data['expenses'] / total_expenses * 100) if total_expenses > 0 else 0
                        lines.append(f"- **{cat}:** ₹{data['expenses']:,.2f} ({pct:.1f}%)")
                return "\n".join(lines)
            return "No category breakdown available. Transactions need to be categorized first."

        else:
            return (
                f"Your total expenses are **₹{total_expenses:,.2f}**. "
                f"If you have a specific question about spending, try asking about "
                f"categories, monthly comparisons, or specific transaction types."
            )

    def _answer_transaction_search(self, question: str,
                                   analysis: AnalysisResult) -> str:
        """Answer transaction search questions."""
        # This would typically search the actual transactions
        # For now, provide summary-based answer

        if any(w in question.lower() for w in ['highest', 'largest', 'maximum']):
            if analysis.unusual_transactions:
                largest = analysis.unusual_transactions[0]
                return (
                    f"Your **largest transaction** appears to be **₹{largest['amount']:,.2f}** "
                    f"on {largest['date']} for {largest['description']}."
                )
            return "I don't have specific transaction details to identify the largest transaction."

        elif any(w in question.lower() for w in ['food', 'dining', 'restaurant', 'swiggy', 'zomato']):
            if analysis.category_breakdown.get('Food and Dining'):
                fd = analysis.category_breakdown['Food and Dining']
                return (
                    f"Your **Food and Dining** expenses total **₹{fd['expenses']:,.2f}** "
                    f"across {fd['count']} transactions."
                )
            return "I don't have food expense data. Please categorize your transactions."

        elif any(w in question.lower() for w in ['shopping', 'amazon', 'online']):
            if analysis.category_breakdown.get('Shopping'):
                shop = analysis.category_breakdown['Shopping']
                return (
                    f"Your **Shopping** expenses total **₹{shop['expenses']:,.2f}** "
                    f"across {shop['count']} transactions."
                )
            return "I don't have shopping expense data."

        else:
            return (
                f"I found {analysis.transaction_count} transactions in your account. "
                f"For specific transaction searches, try uploading your statements and I can "
                f"search through them in detail."
            )

    def _answer_monthly_comparison(self, question: str,
                                   analysis: AnalysisResult) -> str:
        """Answer monthly comparison questions."""
        monthly = analysis.monthly_breakdown

        if not monthly:
            return "I don't have monthly data for comparison. Please upload statements spanning multiple months."

        months = sorted(monthly.keys())

        if len(months) >= 2:
            latest = months[-1]
            prev = months[-2]

            latest_data = monthly[latest]
            prev_data = monthly[prev]

            income_change = latest_data['income'] - prev_data['income']
            expense_change = latest_data['expenses'] - prev_data['expenses']

            income_pct = (income_change / prev_data['income'] * 100) if prev_data['income'] > 0 else 0
            expense_pct = (expense_change / prev_data['expenses'] * 100) if prev_data['expenses'] > 0 else 0

            return (
                f"## {latest} vs {prev}\n\n"
                f"**{latest}:**\n"
                f"- Income: ₹{latest_data['income']:,.2f}\n"
                f"- Expenses: ₹{latest_data['expenses']:,.2f}\n"
                f"- Savings: ₹{latest_data['savings']:,.2f}\n\n"
                f"**{prev}:**\n"
                f"- Income: ₹{prev_data['income']:,.2f}\n"
                f"- Expenses: ₹{prev_data['expenses']:,.2f}\n"
                f"- Savings: ₹{prev_data['savings']:,.2f}\n\n"
                f"**Changes:**\n"
                f"- Income: {'+' if income_change >= 0 else ''}{income_change:,.2f} ({income_pct:+.1f}%)\n"
                f"- Expenses: {'+' if expense_change >= 0 else ''}{expense_change:,.2f} ({expense_pct:+.1f}%)"
            )

        elif len(months) == 1:
            data = monthly[months[0]]
            return (
                f"**{months[0]}:**\n"
                f"- Income: ₹{data['income']:,.2f}\n"
                f"- Expenses: ₹{data['expenses']:,.2f}\n"
                f"- Savings: ₹{data['savings']:,.2f}\n\n"
                f"(Only one month of data available for comparison)"
            )

        return "No monthly data available for comparison."

    def _answer_budget_question(self, question: str,
                                 analysis: AnalysisResult) -> str:
        """Answer budgeting and savings questions."""
        savings_rate = analysis.savings_rate
        total_expenses = analysis.total_expenses
        net_savings = analysis.net_savings

        suggestions = []

        # Generate personalized suggestions
        if savings_rate < 20:
            suggestions.append(
                f"Your savings rate is **{savings_rate:.1f}%**. "
                f"Consider aiming for at least 20% savings. "
                f"That would mean saving approximately ₹{analysis.total_income * 0.2:,.2f} per period."
            )

        if analysis.highest_spending_category:
            cat = analysis.highest_spending_category
            cat_amount = analysis.highest_category_amount
            suggestions.append(
                f"**Reduce {cat} spending:** You're spending ₹{cat_amount:,.2f} on {cat}. "
                f"Try to reduce this by 10-20% by looking for alternatives or setting limits."
            )

        # Find expense categories that could be reduced
        if analysis.category_breakdown:
            entertainment = analysis.category_breakdown.get('Entertainment', {}).get('expenses', 0)
            if entertainment > 0:
                savings_potential = entertainment * 0.3  # 30% potential savings
                suggestions.append(
                    f"**Entertainment audit:** You spend ₹{entertainment:,.2f} on entertainment. "
                    f"Consider reviewing subscriptions and finding free alternatives to save ~₹{savings_potential:,.2f}."
                )

            food_dining = analysis.category_breakdown.get('Food and Dining', {}).get('expenses', 0)
            if food_dining > total_expenses * 0.2:  # If > 20% of expenses
                suggestions.append(
                    f"**Food expenses:** Food & Dining is {food_dining/total_expenses*100:.1f}% of your spending. "
                    f"Consider cooking more at home - even reducing by 20% saves ₹{food_dining*0.2:,.2f}."
                )

        if net_savings < 0:
            suggestions.append(
                "**Warning:** Your expenses exceed your income! "
                "This is unsustainable. Prioritize reducing discretionary spending immediately."
            )

        if not suggestions:
            suggestions.append(
                f"Great financial position! Your savings rate is {savings_rate:.1f}%. "
                f"Consider setting aside some of your ₹{net_savings:,.2f} net savings for emergencies."
            )

        return self._format_budget_response(question, analysis, suggestions)

    def _format_budget_response(self, question: str,
                                 analysis: AnalysisResult,
                                 suggestions: List[str]) -> str:
        """Format budget response."""
        savings_rate = analysis.savings_rate
        net_savings = analysis.net_savings
        total_income = analysis.total_income
        total_expenses = analysis.total_expenses

        response = f"""
## Financial Health Assessment

**Your Financial Snapshot:**
- Total Income: ₹{total_income:,.2f}
- Total Expenses: ₹{total_expenses:,.2f}
- Net Savings: ₹{net_savings:,.2f}
- Savings Rate: {savings_rate:.1f}%

## Recommendations

"""

        for i, suggestion in enumerate(suggestions, 1):
            response += f"{i}. {suggestion}\n\n"

        response += """
## Quick Tips

1. **Track your spending** - Use the transaction view to monitor categories
2. **Set budgets per category** - Especially for discretionary spending
3. **Review subscriptions** - Cancel unused services
4. **Build an emergency fund** - Aim for 3-6 months of expenses
5. **Automate savings** - Set up automatic transfers to savings

*Note: These suggestions are based on your uploaded data and are for informational purposes only.*
"""

        return response

    def _answer_income_question(self, question: str,
                                 analysis: AnalysisResult) -> str:
        """Answer income-related questions."""
        total_income = analysis.total_income

        return (
            f"Your **total income** from uploaded statements is **₹{total_income:,.2f}**. "
            f"This is based on all credit transactions in your account."
        )

    def _answer_general_question(self, question: str,
                                  analysis: AnalysisResult) -> str:
        """Answer general financial questions."""
        return (
            f"Based on your uploaded data:\n\n"
            f"- **Total Income:** ₹{analysis.total_income:,.2f}\n"
            f"- **Total Expenses:** ₹{analysis.total_expenses:,.2f}\n"
            f"- **Net Savings:** ₹{analysis.net_savings:,.2f}\n"
            f"- **Savings Rate:** {analysis.savings_rate:.1f}%\n\n"
            f"You've made {analysis.transaction_count} transactions. "
            f"For specific insights, try asking about categories, monthly comparisons, or budgeting."
        )

    def _generate_rule_based_insights(self, analysis: AnalysisResult) -> AdvisorResponse:
        """Generate insights without LLM."""
        insights = []

        # Basic financial health
        if analysis.savings_rate >= 30:
            insights.append(f"Excellent savings rate of {analysis.savings_rate:.1f}%! You're saving well above the recommended 20%.")
        elif analysis.savings_rate >= 20:
            insights.append(f"Good savings rate of {analysis.savings_rate:.1f}%. You're meeting the recommended 20% savings target.")
        elif analysis.savings_rate >= 10:
            insights.append(f"Your savings rate is {analysis.savings_rate:.1f}%. Consider trying to increase this to at least 20%.")
        elif analysis.savings_rate >= 0:
            insights.append(f"Your savings rate is only {analysis.savings_rate:.1f}%. Try to reduce expenses or increase income.")
        else:
            insights.append(f"Warning: Your expenses exceed income by ₹{-analysis.net_savings:,.2f}. This is unsustainable.")

        # Category insights
        if analysis.highest_spending_category:
            insights.append(
                f"Your highest spending category is **{analysis.highest_spending_category}** "
                f"at ₹{analysis.highest_category_amount:,.2f}."
            )

        # Unusual transactions
        if analysis.unusual_transactions:
            insights.append(
                f"Found {len(analysis.unusual_transactions)} unusual transactions "
                f"(significantly above your average spending)."
            )

        # Monthly trends
        if analysis.month_over_month_changes:
            latest_change = list(analysis.month_over_month_changes.values())[-1]
            if latest_change['percent_change'] > 10:
                insights.append(
                    f"Spending increased by {latest_change['percent_change']:.1f}% compared to previous month."
                )
            elif latest_change['percent_change'] < -10:
                insights.append(
                    f"Great job! Spending decreased by {abs(latest_change['percent_change']):.1f}% compared to previous month."
                )

        answer = "## Key Insights\n\n" + "\n".join(f"- { insight}" for insight in insights)

        return AdvisorResponse(
            answer=answer,
            sources=self._extract_sources(analysis),
            intent="insight_generation",
            confidence="medium",
            suggestions=self._generate_suggestions(analysis),
            warnings=self._generate_warnings(analysis)
        )

    def _generate_suggestions(self, analysis: AnalysisResult) -> List[str]:
        """Generate actionable suggestions."""
        suggestions = []

        # Savings rate suggestions
        if analysis.savings_rate < 20 and analysis.total_income > 0:
            target_savings = analysis.total_income * 0.2
            gap = target_savings - analysis.net_savings
            if gap > 0:
                suggestions.append(
                    f"To reach 20% savings rate, you need to save an additional "
                    f"₹{gap:,.2f} per period."
                )

        # Category-specific suggestions
        if analysis.category_breakdown:
            # Entertainment
            entertainment = analysis.category_breakdown.get('Entertainment', {}).get('expenses', 0)
            if entertainment > 1000:
                suggestions.append(
                    f"Review entertainment subscriptions. You spend ₹{entertainment:,.2f} - "
                    f"consider if all are necessary."
                )

            # Shopping
            shopping = analysis.category_breakdown.get('Shopping', {}).get('expenses', 0)
            if shopping > analysis.total_expenses * 0.2:
                suggestions.append(
                    f"Shopping accounts for {shopping/analysis.total_expenses*100:.1f}% of spending. "
                    f"Consider implementing a 24-hour rule before non-essential purchases."
                )

        # Food suggestions
        food = analysis.category_breakdown.get('Food and Dining', {}).get('expenses', 0)
        if food > analysis.total_expenses * 0.25:
            suggestions.append(
                f"Food & Dining is {food/analysis.total_expenses*100:.1f}% of your spending. "
                f"Consider meal planning and cooking at home more often."
            )

        # Emergency fund reminder
        if analysis.net_savings > 0:
            suggestions.append(
                "Consider building an emergency fund covering 3-6 months of expenses."
            )

        return suggestions[:5]  # Limit to 5 suggestions

    def _generate_warnings(self, analysis: AnalysisResult) -> List[str]:
        """Generate any warnings."""
        warnings = []

        if analysis.net_savings < 0:
            warnings.append("Expenses exceed income - this is financially unsustainable")

        if analysis.savings_rate < 5 and analysis.total_income > 0:
            warnings.append("Very low savings rate - consider reviewing your budget")

        if len(analysis.unusual_transactions) > 3:
            warnings.append(f"Multiple unusual transactions detected - review for potential fraud")

        return warnings

    def _format_monthly_trends(self, monthly_breakdown: Dict) -> str:
        """Format monthly trends for prompt."""
        if not monthly_breakdown:
            return "No monthly data available."

        lines = []
        for month in sorted(monthly_breakdown.keys()):
            data = monthly_breakdown[month]
            lines.append(
                f"{month}: Income ₹{data['income']:,.2f}, "
                f"Expenses ₹{data['expenses']:,.2f}, "
                f"Savings ₹{data['savings']:,.2f}"
            )

        return "\n".join(lines)

    def _extract_sources(self, analysis: AnalysisResult) -> List[Dict]:
        """Extract source references."""
        sources = []

        if analysis.total_income > 0:
            sources.append({
                'type': 'calculation',
                'description': 'Total income from credit transactions',
                'value': analysis.total_income
            })

        if analysis.total_expenses > 0:
            sources.append({
                'type': 'calculation',
                'description': 'Total expenses from debit transactions',
                'value': analysis.total_expenses
            })

        return sources
