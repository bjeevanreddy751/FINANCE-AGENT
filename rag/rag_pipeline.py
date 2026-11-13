"""
RAG Pipeline Module
End-to-end RAG pipeline for financial insights.
"""

from typing import List, Dict, Optional, Any
import json

# LangChain imports
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from rag.retriever import get_context_builder, ContextBuilder, FinancialRetriever
from rag.vector_store import get_vector_store
from prompts.advisor_prompt import (
    advisor_insights_prompt,
    advisor_qa_prompt,
    format_financial_context,
    format_category_breakdown_for_prompt,
    format_unusual_transactions_for_prompt
)
from prompts.intent_prompt import (
    detect_intent_keyword_based,
    route_to_agent,
    INTENTS
)


class FinancialRAGPipeline:
    """
    Complete RAG pipeline for financial Q&A.

    Flow:
    1. Detect user intent
    2. Retrieve relevant context from vector store
    3. Build comprehensive context from transaction data
    4. Generate response using LLM with grounded context
    """

    def __init__(self):
        self.retriever = FinancialRetriever()
        self.context_builder = ContextBuilder()
        self.vector_store = get_vector_store()
        self._llm = None
        self._use_llm = False

    def set_llm(self, llm):
        """Set the LLM for generation."""
        self._llm = llm
        self._use_llm = llm is not None

    def query(self, user_query: str,
              transaction_data: Dict = None,
              analysis_results: Dict = None,
              use_lightweight: bool = False) -> Dict:
        """
        Process a user query through the RAG pipeline.

        Args:
            user_query: The user's question
            transaction_data: Available transaction data
            analysis_results: Pre-computed analysis results
            use_lightweight: If True, use keyword-based intent detection

        Returns:
            Dictionary with answer, context, and metadata
        """
        result = {
            'query': user_query,
            'answer': '',
            'sources': [],
            'intent': None,
            'confidence': 'low',
            'needs_llm': True,
            'raw_response': None
        }

        # Step 1: Detect intent
        if use_lightweight or not self._llm:
            intent_result = detect_intent_keyword_based(user_query)
        else:
            # Would use LLM for intent detection here
            intent_result = detect_intent_keyword_based(user_query)

        result['intent'] = intent_result['intent']
        result['confidence'] = intent_result['confidence']

        # Step 2: Route to appropriate handling
        routing = route_to_agent(
            intent_result['intent'],
            user_query,
            transaction_data.get('summary', {}) if transaction_data else {}
        )

        # Step 3: Build context
        context_parts = []

        # Get RAG context from vector store
        if self.vector_store.get_count() > 0:
            rag_context = self.retriever.retrieve_relevant_context(
                user_query,
                top_k=5
            )
            if rag_context:
                context_parts.append(rag_context)

        # Add data summary
        if analysis_results:
            data_summary = self._format_analysis_for_context(analysis_results)
            context_parts.append(data_summary)

        # Add transaction data if available
        if transaction_data:
            context_parts.append(self._format_transaction_context(transaction_data))

        full_context = "\n\n".join(context_parts) if context_parts else "No financial data available."

        # Step 4: Generate response
        if self._use_llm and self._llm is not None:
            # Use LLM for response generation
            response = self._generate_with_llm(
                user_query,
                full_context,
                analysis_results
            )
            result['answer'] = response
            result['raw_response'] = response
        else:
            # Use rule-based response (no LLM available)
            result['answer'] = self._generate_rule_based_response(
                user_query,
                analysis_results,
                transaction_data
            )
            result['needs_llm'] = True  # Flag that LLM would improve response

        # Step 5: Collect sources
        result['sources'] = self._collect_sources(analysis_results)

        return result

    def _generate_with_llm(self, query: str, context: str,
                           analysis: Dict = None) -> str:
        """
        Generate response using LLM.

        Args:
            query: User's query
            context: Retrieved context
            analysis: Analysis results for additional context

        Returns:
            LLM-generated response
        """
        try:
            # Format the prompt
            prompt = advisor_qa_prompt.format(
                context=context,
                query=query
            )

            # Generate response
            if self._llm:
                response = self._llm.invoke(prompt)
                return response.content if hasattr(response, 'content') else str(response)
            else:
                return self._generate_rule_based_response(query, analysis, None)

        except Exception as e:
            print(f"LLM generation error: {e}")
            return self._generate_rule_based_response(query, analysis, None)

    def _generate_rule_based_response(self, query: str,
                                      analysis: Dict = None,
                                      transaction_data: Dict = None) -> str:
        """
        Generate a rule-based response when LLM is not available.

        Args:
            query: User's query
            analysis: Analysis results
            transaction_data: Transaction data

        Returns:
            Rule-based response
        """
        query_lower = query.lower()

        # Specific categories first: "Show my food expenses" must not fall through to
        # the generic total-expenses answer below.
        category_queries = [
            (('food', 'dining', 'restaurant', 'eating', 'swiggy', 'zomato'), 'Food and Dining'),
            (('grocer', 'supermarket'), 'Groceries'),
            (('shopping', 'amazon', 'flipkart', 'myntra', 'online'), 'Shopping'),
            (('transport', 'uber', 'ola', 'cab', 'fuel', 'petrol'), 'Transportation'),
            (('entertainment', 'netflix', 'spotify', 'movie'), 'Entertainment'),
            (('health', 'hospital', 'pharmacy', 'medical', 'doctor'), 'Healthcare'),
            (('travel', 'flight', 'hotel', 'trip'), 'Travel'),
            (('bill', 'electricity', 'broadband', 'recharge', 'utility'), 'Bills and Utilities'),
            (('rent',), 'Rent'),
        ]

        for keywords, category in category_queries:
            if any(word in query_lower for word in keywords):
                category_data = (analysis or {}).get('category_breakdown', {}).get(category)
                if category_data:
                    spent = category_data.get('expenses', 0)
                    count = category_data.get('count', 0) or 0
                    average = spent / count if count else 0
                    return (
                        f"Your **{category}** spending totals **₹{spent:,.2f}** across "
                        f"{count} transactions (average ₹{average:,.2f} each)."
                    )
                return f"I don't have any **{category}** transactions in the uploaded statements."

        # Budgeting advice
        if any(w in query_lower for w in ['budget', 'tip', 'advice', 'recommend',
                                          'suggest', 'reduce', 'save more', 'how can i save']):
            return self._build_budget_advice(analysis)

        # Handle common query patterns
        if any(w in query_lower for w in ['total income', 'how much income', 'income amount']):
            if analysis and 'total_income' in analysis:
                return f"Based on your uploaded statements, your total income is **₹{analysis['total_income']:,.2f}**."
            return "I don't have enough data to calculate your total income. Please upload your bank statements."

        elif any(w in query_lower for w in ['total expense', 'how much spend', 'total spend', 'expenses']):
            if analysis and 'total_expenses' in analysis:
                return f"Based on your uploaded statements, your total expenses are **₹{analysis['total_expenses']:,.2f}**."
            return "I don't have enough data to calculate your total expenses. Please upload your bank statements."

        elif any(w in query_lower for w in ['saving', 'saved', 'save', 'net saving', 'balance']):
            if analysis and 'net_savings' in analysis:
                savings = analysis['net_savings']
                if savings >= 0:
                    return f"Great! Your net savings are **₹{savings:,.2f}**."
                else:
                    return f"Your net savings are **₹{savings:,.2f}**, indicating expenses exceeded income."
            return "I cannot calculate your savings without uploaded transaction data."

        elif any(w in query_lower for w in ['highest', 'most spend', 'spending most']):
            if analysis and analysis.get('highest_spending_category'):
                return f"Your highest spending category is **{analysis['highest_spending_category']}** with **₹{analysis.get('highest_category_amount', 0):,.2f}**."
            return "I don't have spending category data. Please upload statements and categorize transactions."

        elif any(w in query_lower for w in ['month', 'compare', 'last month', 'previous']):
            if analysis and analysis.get('monthly_breakdown'):
                months = list(analysis['monthly_breakdown'].keys())
                if len(months) >= 2:
                    latest = analysis['monthly_breakdown'][months[-1]]
                    prev = analysis['monthly_breakdown'][months[-2]]
                    return (
                        f"**Month-over-Month Comparison:**\n\n"
                        f"**{months[-1]}:** Income ₹{latest['income']:,.2f}, "
                        f"Expenses ₹{latest['expenses']:,.2f}, "
                        f"Savings ₹{latest['savings']:,.2f}\n\n"
                        f"**{months[-2]}:** Income ₹{prev['income']:,.2f}, "
                        f"Expenses ₹{prev['expenses']:,.2f}, "
                        f"Savings ₹{prev['savings']:,.2f}"
                    )
                elif len(months) == 1:
                    m = analysis['monthly_breakdown'][months[0]]
                    return (f"**{months[0]}:** Income ₹{m['income']:,.2f}, "
                           f"Expenses ₹{m['expenses']:,.2f}, "
                           f"Savings ₹{m['savings']:,.2f}")
            return "I don't have monthly comparison data. Please upload statements spanning multiple months."

        elif any(w in query_lower for w in ['category', 'breakdown', 'where spend', 'spending']):
            if analysis and analysis.get('category_breakdown'):
                breakdown = analysis['category_breakdown']
                lines = ["**Your Spending by Category:**\n"]
                for cat, data in sorted(breakdown.items(), key=lambda x: x[1].get('expenses', 0), reverse=True)[:5]:
                    if data.get('expenses', 0) > 0:
                        lines.append(f"- **{cat}:** ₹{data['expenses']:,.2f}")
                return "\n".join(lines)
            return "I don't have category breakdown data. Please upload and categorize your transactions."

        elif any(w in query_lower for w in ['food', 'dining', 'eating', 'restaurant']):
            if analysis and analysis.get('category_breakdown', {}).get('Food and Dining'):
                fd = analysis['category_breakdown']['Food and Dining']
                return (f"Your **Food and Dining** expenses total **₹{fd.get('expenses', 0):,.2f}** "
                       f"across {fd.get('count', 0)} transactions.")
            return "I don't have food expense data. Please upload statements with food-related transactions."

        elif any(w in query_lower for w in ['shopping', 'amazon', 'online']):
            if analysis and analysis.get('category_breakdown', {}).get('Shopping'):
                shop = analysis['category_breakdown']['Shopping']
                return (f"Your **Shopping** expenses total **₹{shop.get('expenses', 0):,.2f}** "
                       f"across {shop.get('count', 0)} transactions.")
            return "I don't have shopping expense data."

        elif any(w in query_lower for w in ['help', 'what can', 'capabilities', 'do']):
            return (
                "**I can help you analyze your finances!**\n\n"
                "Here are some things you can ask:\n"
                "- *How much did I spend this month?*\n"
                "- *Where am I spending the most?*\n"
                "- *Compare my last two months*\n"
                "- *Show my food expenses*\n"
                "- *Give me suggestions to save money*\n\n"
                "Please upload your bank statements first for personalized insights!"
            )

        else:
            # Default response
            if analysis:
                savings_rate = analysis.get('savings_rate', 0)
                return (
                    f"Based on your uploaded data, I can see you've spent **₹{analysis.get('total_expenses', 0):,.2f}** "
                    f"and earned **₹{analysis.get('total_income', 0):,.2f}**. "
                    f"Your savings rate is **{savings_rate:.1f}%**. "
                    f"For more specific insights, try asking about specific categories or months."
                )
            return (
                "I'd be happy to help with your finances! However, I need you to upload "
                "your bank statements (PDF or CSV) first. Once uploaded, I can help you with:\n"
                "- Spending analysis\n"
                "- Category breakdowns\n"
                "- Monthly comparisons\n"
                "- Savings recommendations\n\n"
                "Try uploading a statement to get started!"
            )

    def _build_budget_advice(self, analysis: Dict = None) -> str:
        """Generate rule-based budgeting recommendations from the analysis."""
        if not analysis:
            return (
                "I need your transactions before I can give budgeting advice. "
                "Upload a statement and I'll point out the categories worth trimming."
            )

        income = analysis.get('total_income', 0)
        expenses = analysis.get('total_expenses', 0)
        savings = analysis.get('net_savings', 0)
        rate = analysis.get('savings_rate', 0)

        lines = ["**Budgeting recommendations:**", ""]

        if income > 0:
            lines.append(
                f"- You keep **{rate:.1f}%** of your income "
                f"(₹{savings:,.2f} saved of ₹{income:,.2f} earned)."
            )
            if rate < 20:
                target = income * 0.2
                lines.append(
                    f"- A 20% savings rate means ₹{target:,.2f}; trimming "
                    f"₹{max(0.0, target - savings):,.2f} of spending gets you there."
                )

        breakdown = analysis.get('category_breakdown') or {}
        top = sorted(
            ((cat, data) for cat, data in breakdown.items() if data.get('expenses', 0) > 0),
            key=lambda item: item[1]['expenses'],
            reverse=True
        )[:3]

        for category, data in top:
            share = (data['expenses'] / expenses * 100) if expenses else 0
            lines.append(
                f"- **{category}** is your largest cost at ₹{data['expenses']:,.2f} "
                f"({share:.0f}% of spending); cutting it 10% saves ₹{data['expenses'] * 0.1:,.2f}."
            )

        if top:
            lines.append("")
            lines.append(f"Start with **{top[0][0]}** - it is the biggest lever on your budget.")

        return "\n".join(lines)

    def _format_analysis_for_context(self, analysis: Dict) -> str:
        """Format analysis results as context string."""
        parts = []

        if not analysis:
            return ""

        if 'total_income' in analysis:
            parts.append(f"Total Income: ₹{analysis['total_income']:,.2f}")

        if 'total_expenses' in analysis:
            parts.append(f"Total Expenses: ₹{analysis['total_expenses']:,.2f}")

        if 'net_savings' in analysis:
            parts.append(f"Net Savings: ₹{analysis['net_savings']:,.2f}")

        if 'savings_rate' in analysis:
            parts.append(f"Savings Rate: {analysis['savings_rate']:.1f}%")

        if analysis.get('highest_spending_category'):
            parts.append(
                f"Highest Spending Category: {analysis['highest_spending_category']} "
                f"(₹{analysis.get('highest_category_amount', 0):,.2f})"
            )

        if analysis.get('monthly_breakdown'):
            parts.append("\nMonthly Breakdown:")
            for month, data in analysis['monthly_breakdown'].items():
                parts.append(
                    f"  {month}: Income ₹{data['income']:,.2f}, "
                    f"Expenses ₹{data['expenses']:,.2f}"
                )

        if analysis.get('category_breakdown'):
            parts.append("\nCategory Spending:")
            for cat, data in list(analysis['category_breakdown'].items())[:5]:
                if data.get('expenses', 0) > 0:
                    parts.append(f"  - {cat}: ₹{data['expenses']:,.2f}")

        return "\n".join(parts)

    def _format_transaction_context(self, data: Dict) -> str:
        """Format transaction data as context."""
        parts = []

        if data.get('summary'):
            parts.append("## Transaction Summary")
            summary = data['summary']
            parts.append(f"Total Transactions: {summary.get('total_transactions', 0)}")
            parts.append(f"Date Range: {summary.get('date_range', 'N/A')}")

        if data.get('recent_transactions'):
            parts.append("\n## Recent Transactions")
            for txn in data['recent_transactions'][:5]:
                parts.append(
                    f"- {txn.get('date', '')}: {txn.get('description', '')} - "
                    f"₹{txn.get('amount', 0):,.2f}"
                )

        return "\n".join(parts)

    def _collect_sources(self, analysis: Dict) -> List[Dict]:
        """Collect source references for the response."""
        sources = []

        if analysis:
            if 'total_income' in analysis:
                sources.append({
                    'type': 'calculation',
                    'description': 'Total income from all credit transactions',
                    'value': analysis['total_income']
                })

            if 'total_expenses' in analysis:
                sources.append({
                    'type': 'calculation',
                    'description': 'Total expenses from all debit transactions',
                    'value': analysis['total_expenses']
                })

            if analysis.get('category_breakdown'):
                for category, data in analysis['category_breakdown'].items():
                    sources.append({
                        'type': 'category',
                        'category': category,
                        'total': data.get('total', 0),
                        'count': data.get('count', 0)
                    })

        return sources

    def add_transactions_to_rag(self, transactions: List[Dict]) -> int:
        """
        Add transactions to the RAG vector store.

        Args:
            transactions: List of transaction dictionaries

        Returns:
            Number of transactions added
        """
        return self.vector_store.add_transactions(transactions)

    def reset_vector_store(self):
        """Reset the vector store."""
        self.vector_store.reset()


# Create LCEL-compatible chain for LangChain integration
def create_rag_chain(llm, retriever: FinancialRetriever = None):
    """
    Create a LangChain LCEL chain for RAG.

    Args:
        llm: LangChain LLM instance
        retriever: FinancialRetriever instance

    Returns:
        LangChain Runnable chain
    """
    if retriever is None:
        retriever = FinancialRetriever()

    def format_context(inputs: Dict) -> Dict:
        """Format context for the LLM."""
        query = inputs.get('query', '')
        analysis = inputs.get('analysis', {})

        context = retriever.retrieve_relevant_context(query, top_k=5)

        if analysis:
            context += "\n\n## Financial Summary\n"
            if 'total_income' in analysis:
                context += f"Total Income: ₹{analysis['total_income']:,.2f}\n"
            if 'total_expenses' in analysis:
                context += f"Total Expenses: ₹{analysis['total_expenses']:,.2f}\n"
            if 'net_savings' in analysis:
                context += f"Net Savings: ₹{analysis['net_savings']:,.2f}\n"

        return {
            'context': context,
            'query': query
        }

    # Create the chain using LCEL
    chain = (
        RunnablePassthrough()
        | RunnableLambda(format_context)
        | advisor_qa_prompt
        | llm
        | StrOutputParser()
    )

    return chain


# Singleton pipeline
_pipeline = None

def get_rag_pipeline() -> FinancialRAGPipeline:
    """Get the singleton RAG pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = FinancialRAGPipeline()
    return _pipeline
