"""
FinanceIQ - Statement-to-Insights Finance Agent
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import base64
import json

from config import config
from agents.orchestrator import AgentOrchestrator, get_orchestrator, reset_orchestrator
from llm_client import get_llm_or_mock, MockLLM

# Page configuration
st.set_page_config(
    page_title="FinanceIQ - AI Finance Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium fintech design
def apply_custom_css():
    """Apply custom styling for the fintech theme."""

    # Plain (non-f) string: the CSS braces must not be parsed as f-string fields.
    st.markdown("""
    <style>
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: #1e293b;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #334155;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.1);
    }

    .income-value {
        color: #22c55e;
        font-size: 2rem;
        font-weight: bold;
    }

    .expense-value {
        color: #ef4444;
        font-size: 2rem;
        font-weight: bold;
    }

    .primary-value {
        color: #3b82f6;
        font-size: 2rem;
        font-weight: bold;
    }

    .section-title {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3b82f6;
    }

    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }

    .upload-box {
        background: #1e293b;
        border: 2px dashed #334155;
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .upload-box:hover {
        border-color: #3b82f6;
        background: rgba(59, 130, 246, 0.05);
    }

    .chat-message {
        padding: 1rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
    }

    .chat-user {
        background: #3b82f6;
        color: white;
        margin-left: 2rem;
    }

    .chat-assistant {
        background: #1e293b;
        border: 1px solid #334155;
        margin-right: 2rem;
    }

    .stDataFrame {
        background: #1e293b !important;
        border-radius: 0.5rem;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #0f172a;
    }

    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #3b82f6;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] button [role="icon"] svg {
        color: #94a3b8;
    }

    .stTabs [data-baseweb="tab"] button[aria-selected="true"] {
        color: #3b82f6 !important;
    }

    .footer-text {
        color: #94a3b8;
        font-size: 0.875rem;
        text-align: center;
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)


def format_currency(amount: float) -> str:
    """Format amount as currency string."""
    if amount >= 0:
        return f"₹{amount:,.2f}"
    else:
        return f"-₹{abs(amount):,.2f}"


def format_percentage(value: float) -> str:
    """Format as percentage string."""
    return f"{value:.1f}%"


def render_dashboard(orchestrator: AgentOrchestrator):
    """Render the Dashboard page."""
    st.markdown('<h1 class="app-header">📊 Finance Dashboard</h1>', unsafe_allow_html=True)

    analysis = orchestrator.get_analysis()

    if analysis is None or analysis.transaction_count == 0:
        st.info("👋 Welcome to FinanceIQ! Upload your bank statements to see your financial dashboard.")
        st.markdown("""
        ### Get Started
        1. Go to **Upload Statements** to add your bank statements (PDF or CSV)
        2. Your transactions will be automatically extracted and categorized
        3. View insights on the **Dashboard** and **Financial Analytics** pages
        4. Ask questions to the **AI Financial Assistant**

        *No data is stored permanently. Your privacy is protected.*
        """, unsafe_allow_html=True)
        return

    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="        color: #94a3b8; font-size: 0.875rem;">Total Balance</div>
            <div class="primary-value">{format_currency(analysis.net_savings)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="        color: #94a3b8; font-size: 0.875rem;">Total Income</div>
            <div class="income-value">{format_currency(analysis.total_income)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="        color: #94a3b8; font-size: 0.875rem;">Total Expenses</div>
            <div class="expense-value">{format_currency(analysis.total_expenses)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div style="        color: #94a3b8; font-size: 0.875rem;">Savings Rate</div>
            <div class="income-value">{format_percentage(analysis.savings_rate)}</div>
        </div>
        """, unsafe_allow_html=True)

    # Charts Section
    st.markdown('<h2 class="section-title">📈 Financial Analysis</h2>', unsafe_allow_html=True)

    # Monthly Income vs Expenses Chart
    monthly_df = orchestrator.get_monthly_data()

    if not monthly_df.empty:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Monthly Income vs Expenses", "Expense Category Breakdown"),
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )

        # Bar chart
        fig.add_trace(
            go.Bar(name="Income", x=monthly_df['month'], y=monthly_df['income'],
                   marker_color="#22c55e", hovertemplate="Income: ₹%{y:,.0f}<extra></extra>"),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name="Expenses", x=monthly_df['month'], y=monthly_df['expenses'],
                   marker_color="#ef4444", hovertemplate="Expenses: ₹%{y:,.0f}<extra></extra>"),
            row=1, col=1
        )

        # Pie chart
        categories, amounts = orchestrator.get_category_chart_data()
        if categories and amounts:
            fig.add_trace(
                go.Pie(labels=categories, values=amounts,
                       marker_colors=px.colors.qualitative.Set3),
                row=1, col=2
            )

        fig.update_layout(
            height=400,
            showlegend=True,
            plot_bgcolor="#0f172a",
            paper_bgcolor="#0f172a",
            font=dict(color="#ffffff"),
            xaxis=dict(title="Month", tickfont=dict(color="#94a3b8")),
            yaxis=dict(title="Amount (₹)", tickfont=dict(color="#94a3b8")),
            xaxis2=dict(tickfont=dict(color="#94a3b8")),
            yaxis2=dict(showticklabels=False),
        )

        st.plotly_chart(fig, use_container_width=True)

    # Recent Transactions
    st.markdown('<h2 class="section-title">🕐 Recent Transactions</h2>', unsafe_allow_html=True)

    if analysis.recent_transactions:
        recent_df = pd.DataFrame(analysis.recent_transactions)
        recent_df = recent_df.sort_values('date', ascending=False)

        # Display as table with custom styling
        st.dataframe(
            recent_df[['date', 'description', 'amount', 'category']],
            column_config={
                "date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                "description": st.column_config.TextColumn("Description", width="large"),
                "amount": st.column_config.NumberColumn("Amount", format="₹%.2f"),
                "category": st.column_config.TextColumn("Category", width="medium"),
            },
            hide_index=True,
            use_container_width=True
        )

    # Transaction Stats
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Transactions", analysis.transaction_count)

    with col2:
        st.metric("Date Range", analysis.date_range)

    with col3:
        st.metric("Avg Daily Spending", format_currency(analysis.average_daily_expenditure))

    # Unusual Transactions Alert
    if analysis.unusual_transactions:
        st.markdown('<h2 class="section-title">⚠️ Unusual Transactions</h2>', unsafe_allow_html=True)
        with st.expander("View unusual transactions"):
            for txn in analysis.unusual_transactions[:5]:
                st.warning(f"""
                **{txn['date']}** - {txn['description']}
                - Amount: **{format_currency(txn['amount'])}**
                - Reason: {txn['reason']}
                """)


def render_upload_page(orchestrator: AgentOrchestrator):
    """Render the Upload Statements page."""
    st.markdown('<h1 class="app-header">📄 Upload Bank Statements</h1>', unsafe_allow_html=True)

    st.markdown("""
    Upload your bank statements in PDF or CSV format. FinanceIQ will automatically:
    - Extract all transactions
    - Categorize expenses
    - Analyze your financial patterns
    - Index data for AI-powered search
    """, unsafe_allow_html=True)

    # File upload section
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        st.markdown("### 📑 PDF Statement")
        pdf_file = st.file_uploader(
            "Upload PDF Bank Statement",
            type=["pdf"],
            help="Upload your bank statement PDF file"
        )

        if pdf_file:
            st.success(f"✅ **{pdf_file.name}** selected ({pdf_file.size / 1024:.1f} KB)")

            if st.button("Process PDF Statement", type="primary"):
                with st.spinner("Extracting transactions from PDF..."):
                    result = orchestrator.process_statement(
                        pdf_file.read(),
                        "pdf",
                        pdf_file.name
                    )

                    if result.get('success'):
                        st.success(f"✅ Successfully processed {result['stats'].get('valid', 0)} transactions!")
                        st.balloons()
                    else:
                        st.error(f"❌ Error: {result.get('error', 'Unknown error')}")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        st.markdown("### 📊 CSV Statement")
        csv_file = st.file_uploader(
            "Upload CSV Bank Statement",
            type=["csv"],
            help="Upload your bank statement CSV file"
        )

        if csv_file:
            st.success(f"✅ **{csv_file.name}** selected ({csv_file.size / 1024:.1f} KB)")

            # Show preview
            try:
                df_preview = pd.read_csv(csv_file)
                st.dataframe(df_preview.head(10), use_container_width=True, hide_index=True)
            except Exception:
                pass

            if st.button("Process CSV Statement", type="primary"):
                csv_file.seek(0)
                with st.spinner("Processing CSV transactions..."):
                    result = orchestrator.process_statement(
                        csv_file.read(),
                        "csv",
                        csv_file.name
                    )

                    if result.get('success'):
                        st.success(f"✅ Successfully processed {result['stats'].get('valid', 0)} transactions!")
                        st.balloons()
                    else:
                        st.error(f"❌ Error: {result.get('error', 'Unknown error')}")

        st.markdown('</div>', unsafe_allow_html=True)

    # Sample Data Button
    st.markdown("---")
    st.markdown('<h3 class="section-title">🎯 Try with Sample Data</h3>', unsafe_allow_html=True)

    st.info("""
    **Not ready to upload your own statements?** Try FinanceIQ with sample data first!
    """)

    if st.button("Load Sample Transactions", type="secondary"):
        with st.spinner("Loading sample data..."):
            result = orchestrator.process_sample_data()

            if result.get('success'):
                st.success("✅ Sample data loaded successfully!")
                st.info(f"""
                **Sample Data Summary:**
                - Transactions: {result['stats'].get('total_found', 0)}
                - Date Range: {result['stats'].get('date_range', 'N/A')}

                Explore the Dashboard and Analytics to see FinanceIQ in action!
                """)
                st.balloons()
            else:
                st.error(f"❌ Could not load sample data: {result.get('error')}")

    # Processing History
    if orchestrator.processing_history:
        st.markdown('<h3 class="section-title">📋 Processing History</h3>', unsafe_allow_html=True)

        history_df = pd.DataFrame(orchestrator.processing_history)
        st.dataframe(
            history_df[['filename', 'file_type', 'timestamp', 'transaction_count']],
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Processed", format="DD MMM YYYY, HH:mm"),
            },
            hide_index=True,
            use_container_width=True
        )

    # Clear Data Button
    st.markdown("---")
    confirm_reset = st.checkbox("Yes, delete all uploaded transactions and analysis")

    if st.button("🗑️ Clear All Data and Reset", type="secondary", help="Delete all uploaded data and reset the session"):
        if confirm_reset:
            reset_orchestrator()
            st.success("✅ All data cleared. Starting fresh!")
            st.rerun()
        else:
            st.warning("Tick the confirmation box above before clearing your data.")


def render_transactions_page(orchestrator: AgentOrchestrator):
    """Render the Transactions page."""
    st.markdown('<h1 class="app-header">💳 Transactions</h1>', unsafe_allow_html=True)

    df = orchestrator.get_transactions_df()

    if df.empty:
        st.info("📭 No transactions yet. Upload a bank statement to see your transactions here.")
        return

    # Add category column if missing
    if 'category' not in df.columns:
        df['category'] = 'Uncategorized'

    # Filters
    st.markdown('<h3 class="section-title">🔍 Filter Transactions</h3>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        search_query = st.text_input("Search transactions", placeholder="Search by description...")

    with col2:
        categories = ['All'] + [c for c in df['category'].unique() if pd.notna(c)]
        selected_category = st.selectbox("Filter by category", categories)

    with col3:
        min_amount = st.number_input("Min amount", min_value=0.0, value=0.0, step=100.0)

    with col4:
        amount_ceiling = float(df['amount'].max()) if 'amount' in df.columns else 0.0
        amount_ceiling = max(1000.0, ((amount_ceiling // 1000) + 1) * 1000)
        max_amount = st.number_input(
            "Max amount",
            min_value=0.0,
            value=amount_ceiling,
            step=100.0
        )

    # Apply filters
    filtered_df = df.copy()

    if search_query:
        filtered_df = filtered_df[
            filtered_df['description'].str.lower().str.contains(search_query.lower(), na=False)
        ]

    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['category'] == selected_category]

    filtered_df = filtered_df[
        (filtered_df['amount'] >= min_amount) &
        (filtered_df['amount'] <= max_amount)
    ]

    # Sort by date
    if 'date' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('date', ascending=False)

    # Display filtered transactions
    st.markdown(f"**Showing {len(filtered_df)} transactions**")

    if len(filtered_df) > 0:
        st.dataframe(
            filtered_df,
            column_config={
                "date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                "description": st.column_config.TextColumn("Description", width="large"),
                "amount": st.column_config.NumberColumn("Amount", format="₹%.2f"),
                "debit": st.column_config.NumberColumn("Debit", format="₹%.2f"),
                "credit": st.column_config.NumberColumn("Credit", format="₹%.2f"),
                "category": st.column_config.TextColumn("Category", width="medium"),
            },
            hide_index=True,
            use_container_width=True
        )

        # Edit Categories Section
        st.markdown('<h3 class="section-title">✏️ Edit Categories</h3>', unsafe_allow_html=True)

        st.info("Click on a transaction row below to edit its category.")

        # Category edit interface
        edit_col1, edit_col2 = st.columns([3, 1])

        with edit_col1:
            edit_index = st.number_input(
                "Transaction index to edit",
                min_value=0,
                max_value=len(df) - 1,
                value=0
            )

        with edit_col2:
            new_category = st.selectbox(
                "New category",
                config.CATEGORIES,
                index=config.CATEGORIES.index(df.iloc[edit_index]['category']) if edit_index < len(df) and df.iloc[edit_index].get('category') in config.CATEGORIES else 0
            )

        current_txn = df.iloc[edit_index] if edit_index < len(df) else None

        if current_txn is not None:
            st.write(f"Editing transaction #{edit_index}: **{current_txn.get('description', 'N/A')}**")

            if st.button("Update Category", type="primary"):
                success = orchestrator.update_category(edit_index, new_category)
                if success:
                    st.success(f"✅ Category updated to **{new_category}**")
                    st.rerun()
                else:
                    st.error("❌ Failed to update category")

        # Export Options
        st.markdown('<h3 class="section-title">📤 Export Data</h3>', unsafe_allow_html=True)

        export_col1, export_col2, export_col3 = st.columns(3)

        with export_col1:
            if st.button("📥 Export All as CSV"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    "transactions.csv",
                    "text/csv",
                    key="export_all"
                )

        with export_col2:
            if st.button("📥 Export Filtered as CSV"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    "filtered_transactions.csv",
                    "text/csv",
                    key="export_filtered"
                )

        with export_col3:
            if st.button("📋 Copy to Clipboard"):
                st.code(filtered_df.to_csv(index=False), language="csv")
                st.success("Copied to clipboard (see code below)")


def render_analytics_page(orchestrator: AgentOrchestrator):
    """Render the Financial Analytics page."""
    st.markdown('<h1 class="app-header">📈 Financial Analytics</h1>', unsafe_allow_html=True)

    analysis = orchestrator.get_analysis()

    if analysis is None or analysis.transaction_count == 0:
        st.info("📊 Upload bank statements to see detailed financial analytics.")
        return

    # Monthly Overview
    st.markdown('<h2 class="section-title">📅 Monthly Overview</h2>', unsafe_allow_html=True)

    monthly_df = orchestrator.get_monthly_data()

    if not monthly_df.empty:
        # Income vs Expenses Chart
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Income vs Expenses by Month", " Savings Trend")
        )

        # Bar chart for income/expenses
        fig.add_trace(
            go.Bar(name="Income", x=monthly_df['month'], y=monthly_df['income'],
                   marker_color="#22c55e"),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name="Expenses", x=monthly_df['month'], y=monthly_df['expenses'],
                   marker_color="#ef4444"),
            row=1, col=1
        )

        # Line chart for savings
        fig.add_trace(
            go.Scatter(
                name="Savings",
                x=monthly_df['month'],
                y=monthly_df['savings'],
                mode='lines+markers',
                line=dict(color="#3b82f6", width=3),
                marker=dict(size=8, color="#3b82f6")
            ),
            row=2, col=1
        )

        fig.update_layout(
            height=500,
            showlegend=True,
            plot_bgcolor="#0f172a",
            paper_bgcolor="#0f172a",
            font=dict(color="#ffffff"),
            hovermode='x unified'
        )

        fig.update_xaxes(
            tickfont=dict(color="#94a3b8"),
            gridcolor="#334155"
        )
        fig.update_yaxes(
            tickfont=dict(color="#94a3b8"),
            gridcolor="#334155"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Monthly Data Table
        st.dataframe(
            monthly_df,
            column_config={
                "month": st.column_config.TextColumn("Month"),
                "income": st.column_config.NumberColumn("Income", format="₹%.0f"),
                "expenses": st.column_config.NumberColumn("Expenses", format="₹%.0f"),
                "savings": st.column_config.NumberColumn("Savings", format="₹%.0f"),
            },
            hide_index=True,
            use_container_width=True
        )

    # Category Analysis
    st.markdown('<h2 class="section-title">📊 Category Analysis</h2>', unsafe_allow_html=True)

    if analysis.category_breakdown:
        categories, amounts = orchestrator.get_category_chart_data()

        if categories:
            # Horizontal bar chart
            fig = px.bar(
                x=amounts,
                y=categories,
                orientation='h',
                title="Expenses by Category",
                color=amounts,
                color_continuous_scale='RdYlGn_r',
                labels={'x': 'Amount (₹)', 'y': 'Category'}
            )

            fig.update_layout(
                height=400,
                plot_bgcolor="#0f172a",
                paper_bgcolor="#0f172a",
                font=dict(color="#ffffff"),
                xaxis=dict(tickfont=dict(color="#94a3b8")),
                yaxis=dict(tickfont=dict(color="#94a3b8")),
                coloraxis_showscale=False
            )

            st.plotly_chart(fig, use_container_width=True)

            # Category Table
            cat_data = []
            for cat, data in sorted(analysis.category_breakdown.items(),
                                    key=lambda x: x[1].get('expenses', 0), reverse=True):
                if data.get('expenses', 0) > 0:
                    cat_data.append({
                        'Category': cat,
                        'Total Spent': data['expenses'],
                        'Income Received': data['income'],
                        'Transactions': data['count'],
                        '% of Total': (data['expenses'] / analysis.total_expenses * 100) if analysis.total_expenses > 0 else 0
                    })

            if cat_data:
                cat_df = pd.DataFrame(cat_data)
                st.dataframe(
                    cat_df,
                    column_config={
                        "Category": st.column_config.TextColumn("Category", width="large"),
                        "Total Spent": st.column_config.NumberColumn("Total Spent", format="₹%.2f"),
                        "Income Received": st.column_config.NumberColumn("Income", format="₹%.2f"),
                        "Transactions": st.column_config.NumberColumn("Transactions"),
                        "% of Total": st.column_config.NumberColumn("% of Total", format="%.1f%%"),
                    },
                    hide_index=True,
                    use_container_width=True
                )

    # Insights Section
    st.markdown('<h2 class="section-title">💡 Key Insights</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Highest Spending Category",
                  analysis.highest_spending_category or "N/A",
                  format_currency(analysis.highest_category_amount))

    with col2:
        st.metric("Average Daily Spending",
                  format_currency(analysis.average_daily_expenditure))

    # Month-over-Month Changes
    if analysis.month_over_month_changes:
        st.markdown('<h3 class="section-title">📉 Month-over-Month Changes</h3>', unsafe_allow_html=True)

        changes_df = pd.DataFrame([
            {
                'Month': month,
                'Previous Expenses': data['previous_expenses'],
                'Current Expenses': data['current_expenses'],
                'Change': data['absolute_change'],
                'Change %': data['percent_change'],
                'Trend': data['trend'].capitalize()
            }
            for month, data in analysis.month_over_month_changes.items()
        ])

        if not changes_df.empty:
            fig = px.bar(
                changes_df,
                x='Month',
                y='Change',
                color='Trend',
                color_discrete_map={'Increase': '#ef4444', 'Decrease': '#22c55e'},
                title="Month-over-Month Expense Changes",
                labels={'Month': 'Month', 'Change': 'Change Amount (₹)'}
            )

            fig.update_layout(
                height=300,
                plot_bgcolor="#0f172a",
                paper_bgcolor="#0f172a",
                font=dict(color="#ffffff"),
            )

            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                changes_df,
                column_config={
                    "Month": st.column_config.TextColumn("Month"),
                    "Previous Expenses": st.column_config.NumberColumn("Prev Expenses", format="₹%.0f"),
                    "Current Expenses": st.column_config.NumberColumn("Curr Expenses", format="₹%.0f"),
                    "Change": st.column_config.NumberColumn("Change", format="%+.0f"),
                    "Change %": st.column_config.NumberColumn("Change %", format="%+.1f%%"),
                    "Trend": st.column_config.TextColumn("Trend"),
                },
                hide_index=True,
                use_container_width=True
            )

    # Unusual Transactions
    if analysis.unusual_transactions:
        st.markdown('<h2 class="section-title">⚠️ Unusual Transactions</h2>', unsafe_allow_html=True)
        st.warning(f"Found **{len(analysis.unusual_transactions)}** transactions that are significantly above your average spending.")

        unusual_data = []
        for txn in analysis.unusual_transactions:
            unusual_data.append({
                'Date': txn['date'],
                'Description': txn['description'],
                'Amount': txn['amount'],
                'Reason': txn['reason']
            })

        unusual_df = pd.DataFrame(unusual_data)
        st.dataframe(
            unusual_df,
            column_config={
                "Date": st.column_config.TextColumn("Date"),
                "Description": st.column_config.TextColumn("Description", width="large"),
                "Amount": st.column_config.NumberColumn("Amount", format="₹%.2f"),
                "Reason": st.column_config.TextColumn("Reason", width="large"),
            },
            hide_index=True,
            use_container_width=True
        )


def answer_finance_question(orchestrator: AgentOrchestrator, query: str) -> str:
    """Ask the FinanceIQ agents a question and return the answer text."""
    analysis = orchestrator.get_analysis()

    if analysis is None or analysis.transaction_count == 0:
        return (
            "I don't have any financial data to answer that yet.\n\n"
            "**Upload your statements first:**\n"
            "1. Go to the **Upload** page\n"
            "2. Upload a PDF/CSV statement, or click **Load Sample Transactions**\n"
            "3. Come back here and ask again"
        )

    try:
        result = orchestrator.query(query)
    except Exception as exc:
        return f"Something went wrong while analysing your data: {exc}"

    answer = (result.get('answer') or '').strip()
    if not answer:
        return (
            "I could not work out an answer from your data. "
            "Try asking about your spending, a category, or a month."
        )
    return answer


def render_assistant_page(orchestrator: AgentOrchestrator):
    """Render the AI Financial Assistant page."""
    st.markdown('<h1 class="app-header">🤖 AI Financial Assistant</h1>', unsafe_allow_html=True)

    st.markdown("""
    Ask questions about your finances and get AI-powered answers based on your uploaded data.

    ### Example Questions
    - *How much did I spend this month?*
    - *Where am I spending the most?*
    - *What was my highest transaction?*
    - *Compare my last two months*
    - *How much did I save?*
    - *Give me suggestions to reduce expenses*
    - *Show my food expenses*
    """, unsafe_allow_html=True)

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Chat interface
    st.markdown('<h3 class="section-title">💬 Chat with Your Finance AI</h3>', unsafe_allow_html=True)

    # Display chat history (st.chat_message keeps Markdown formatting intact)
    if not st.session_state.messages:
        st.info("Ask a question below, or tap one of the quick questions at the bottom")

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Chat input
    if query := st.chat_input("Ask me anything about your finances..."):
        st.session_state.messages.append({'role': 'user', 'content': query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Analysing your finances..."):
            response = answer_finance_question(orchestrator, query)

        st.session_state.messages.append({'role': 'assistant', 'content': response})

        with st.chat_message("assistant"):
            st.markdown(response)

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History", type="secondary"):
            st.session_state.messages = []
            st.rerun()

    # Quick question buttons
    st.markdown('<h3 class="section-title">⚡ Quick Questions</h3>', unsafe_allow_html=True)

    quick_questions = [
        "How much did I spend?",
        "Where am I spending the most?",
        "Show my food expenses",
        "Compare last two months",
        "How much did I save?",
        "Give me budgeting tips"
    ]

    cols = st.columns(3)
    for i, question in enumerate(quick_questions):
        if cols[i % 3].button(question, key=f"quick_{i}"):
            st.session_state.messages.append({'role': 'user', 'content': question})
            with st.spinner("Analysing your finances..."):
                answer = answer_finance_question(orchestrator, question)
            st.session_state.messages.append({'role': 'assistant', 'content': answer})
            st.rerun()


def render_reports_page(orchestrator: AgentOrchestrator):
    """Render the Reports page."""
    st.markdown('<h1 class="app-header">📋 Financial Reports</h1>', unsafe_allow_html=True)

    analysis = orchestrator.get_analysis()

    if analysis is None or analysis.transaction_count == 0:
        st.info("📊 Upload bank statements to generate financial reports.")
        return

    # Generate Report
    st.markdown('<h2 class="section-title">📄 Financial Summary Report</h2>', unsafe_allow_html=True)

    # Report header
    st.markdown(f"""
    <div style="background: #1e293b; padding: 1.5rem; border-radius: 0.75rem; margin-bottom: 1.5rem;">
        <h3 style="        color: #ffffff; margin-top: 0;">FinanceIQ Financial Report</h3>
        <p style="        color: #94a3b8;">
            Generated on: {pd.Timestamp.now().strftime('%d %B %Y, %H:%M')}<br>
            Data Range: {analysis.date_range}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Executive Summary
    st.markdown('<h3 class="section-title">Executive Summary</h3>', unsafe_allow_html=True)

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.metric("Total Income", format_currency(analysis.total_income))

    with summary_col2:
        st.metric("Total Expenses", format_currency(analysis.total_expenses))

    with summary_col3:
        st.metric("Net Savings", format_currency(analysis.net_savings))

    # Financial Health Indicators
    st.markdown('<h3 class="section-title">Financial Health Indicators</h3>', unsafe_allow_html=True)

    health_data = []

    # Savings Rate
    savings_rate = analysis.savings_rate
    health_data.append({
        'Indicator': 'Savings Rate',
        'Value': f"{savings_rate:.1f}%",
        'Status': 'Excellent' if savings_rate >= 30 else 'Good' if savings_rate >= 20 else 'Fair' if savings_rate >= 10 else 'Needs Improvement',
        'Target': '≥ 20%'
    })

    # Expense Ratio
    expense_ratio = (analysis.total_expenses / analysis.total_income * 100) if analysis.total_income > 0 else 0
    health_data.append({
        'Indicator': 'Expense Ratio',
        'Value': f"{expense_ratio:.1f}%",
        'Status': 'Good' if expense_ratio < 70 else 'Fair' if expense_ratio < 85 else 'High',
        'Target': '< 80%'
    })

    # Transaction Volume
    health_data.append({
        'Indicator': 'Transaction Volume',
        'Value': analysis.transaction_count,
        'Status': 'Active',
        'Target': 'N/A'
    })

    # Average Transaction
    avg_txn = analysis.total_expenses / analysis.transaction_count if analysis.transaction_count > 0 else 0
    health_data.append({
        'Indicator': 'Avg Transaction Size',
        'Value': format_currency(avg_txn),
        'Status': 'Normal',
        'Target': 'N/A'
    })

    health_df = pd.DataFrame(health_data)
    st.dataframe(
        health_df,
        column_config={
            "Indicator": st.column_config.TextColumn("Indicator", width="large"),
            "Value": st.column_config.TextColumn("Current Value", width="medium"),
            "Status": st.column_config.TextColumn("Status", width="medium"),
            "Target": st.column_config.TextColumn("Recommended Target", width="medium"),
        },
        hide_index=True,
        use_container_width=True
    )

    # Monthly Breakdown in Report
    st.markdown('<h3 class="section-title">Monthly Breakdown</h3>', unsafe_allow_html=True)

    monthly_df = orchestrator.get_monthly_data()

    if not monthly_df.empty:
        monthly_report = monthly_df.copy()
        monthly_report['Savings Rate'] = (monthly_report['savings'] / monthly_report['income'] * 100).apply(lambda x: f"{x:.1f}%")
        monthly_report['Income'] = monthly_report['income'].apply(lambda x: format_currency(x))
        monthly_report['Expenses'] = monthly_report['expenses'].apply(lambda x: format_currency(x))
        monthly_report['Savings'] = monthly_report['savings'].apply(lambda x: format_currency(x))

        st.dataframe(
            monthly_report[['month', 'Income', 'Expenses', 'Savings', 'Savings Rate']],
            column_config={
                "month": st.column_config.TextColumn("Month"),
                "Income": st.column_config.TextColumn("Income"),
                "Expenses": st.column_config.TextColumn("Expenses"),
                "Savings": st.column_config.TextColumn("Savings"),
                "Savings Rate": st.column_config.TextColumn("Savings Rate"),
            },
            hide_index=True,
            use_container_width=True
        )

    # Category Summary
    st.markdown('<h3 class="section-title">Top Spending Categories</h3>', unsafe_allow_html=True)

    if analysis.category_breakdown:
        top_categories = sorted(
            analysis.category_breakdown.items(),
            key=lambda x: x[1].get('expenses', 0),
            reverse=True
        )[:5]

        cat_report_data = []
        for cat, data in top_categories:
            if data.get('expenses', 0) > 0:
                cat_report_data.append({
                    'Category': cat,
                    'Amount': format_currency(data['expenses']),
                    'Transactions': data['count'],
                    'Pct of Total': f"{(data['expenses'] / analysis.total_expenses * 100):.1f}%" if analysis.total_expenses > 0 else "0%"
                })

        if cat_report_data:
            cat_report_df = pd.DataFrame(cat_report_data)
            st.dataframe(
                cat_report_df,
                column_config={
                    "Category": st.column_config.TextColumn("Category", width="large"),
                    "Amount": st.column_config.TextColumn("Amount"),
                    "Transactions": st.column_config.NumberColumn("Transactions"),
                    "Pct of Total": st.column_config.TextColumn("% of Total"),
                },
                hide_index=True,
                use_container_width=True
            )

    # Download Options
    st.markdown('<h2 class="section-title">⬇️ Download Reports</h2>', unsafe_allow_html=True)

    download_col1, download_col2, download_col3, download_col4 = st.columns(4)

    # Generate CSV
    transactions_df = orchestrator.get_transactions_df()

    if not transactions_df.empty:
        csv_data = transactions_df.to_csv(index=False)

        with download_col1:
            st.download_button(
                "📥 Download Transactions CSV",
                csv_data,
                "financeiq_transactions.csv",
                "text/csv",
                key="download_csv"
            )

        with download_col2:
            st.download_button(
                "📥 Download Summary JSON",
                json.dumps(orchestrator.get_summary(), indent=2, default=str),
                "financeiq_summary.json",
                "application/json",
                key="download_json"
            )

        with download_col3:
            st.download_button(
                "📥 Download Full Report (TXT)",
                f"""FINANCEIQ FINANCIAL REPORT
==========================
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Data Range: {analysis.date_range}

EXECUTIVE SUMMARY
-----------------
Total Income: {format_currency(analysis.total_income)}
Total Expenses: {format_currency(analysis.total_expenses)}
Net Savings: {format_currency(analysis.net_savings)}
Savings Rate: {format_percentage(analysis.savings_rate)}
Transaction Count: {analysis.transaction_count}

TOP CATEGORIES
--------------
{chr(10).join([f"- {cat}: {format_currency(data['expenses'])} ({data['count']} txns)" for cat, data in sorted(analysis.category_breakdown.items(), key=lambda x: x[1].get('expenses', 0), reverse=True)[:5] if data.get('expenses', 0) > 0])}

MONTHLY OVERVIEW
----------------
{chr(10).join([f"{month}: Income {format_currency(data['income'])}, Expenses {format_currency(data['expenses'])}, Savings {format_currency(data['savings'])}" for month, data in analysis.monthly_breakdown.items()])}

 generated by FinanceIQ
""",
                "financeiq_report.txt",
                "text/plain",
                key="download_report"
            )

        with download_col4:
            st.download_button(
                "📥 Sample Data (PDF Guide)",
                "Sample bank statement format guide - See documentation for details",
                "format_guide.txt",
                "text/plain",
                key="download_guide",
                disabled=True
            )

    # Print/Export Instructions
    st.markdown('<h3 class="section-title">📑 Export Options</h3>', unsafe_allow_html=True)

    st.markdown("""
    **Available Export Formats:**

    | Format | Description | Best For |
    |--------|-------------|----------|
    | **CSV** | Raw transaction data | Excel, Google Sheets, further analysis |
    | **JSON** | Structured summary data | API integration, programmatic use |
    | **Text Report** | Human-readable summary | Printing, email, quick review |

    **Tips:**
    - Use CSV export to import data into spreadsheet software
    - JSON format preserves all calculated metrics
    - Text report is formatted for printing or sharing
    """, unsafe_allow_html=True)


def main():
    """Main application entry point."""
    apply_custom_css()

    # Initialize session state for orchestrator
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = get_orchestrator()

    orchestrator = st.session_state.orchestrator

    # Sidebar navigation
    st.sidebar.title("📊 FinanceIQ")
    st.sidebar.markdown("### AI-Powered Finance Agent")

    # Navigation tabs
    pages = {
        "📊 Dashboard": "dashboard",
        "📄 Upload": "upload",
        "💳 Transactions": "transactions",
        "📈 Analytics": "analytics",
        "🤖 Assistant": "assistant",
        "📋 Reports": "reports"
    }

    selected_page = st.sidebar.radio(
        "Navigation",
        list(pages.keys()),
        format_func=lambda x: x.split(' ', 1)[1] if ' ' in x else x
    )

    # Quick Stats in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Stats")

    analysis = orchestrator.get_analysis()
    if analysis and analysis.transaction_count > 0:
        sidebar_stats = [
            ("Income", format_currency(analysis.total_income), "#22c55e"),
            ("Expenses", format_currency(analysis.total_expenses), "#ef4444"),
            ("Savings", format_currency(analysis.net_savings), "#3b82f6"),
        ]

        for label, value, color in sidebar_stats:
            st.sidebar.markdown(f"""
            <div style="background: #1e293b; padding: 0.75rem; border-radius: 0.5rem; margin-bottom: 0.5rem;">
                <div style="        color: #94a3b8; font-size: 0.75rem;">{label}</div>
                <div style="        color: {color}; font-size: 1.125rem; font-weight: 600;">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    # Footer in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div style="        color: #94a3b8; font-size: 0.75rem; text-align: center;">
        FinanceIQ v1.0<br>
        Powered by Agentic AI
    </div>
    """, unsafe_allow_html=True)

    # Render selected page
    if selected_page == "📊 Dashboard":
        render_dashboard(orchestrator)
    elif selected_page == "📄 Upload":
        render_upload_page(orchestrator)
    elif selected_page == "💳 Transactions":
        render_transactions_page(orchestrator)
    elif selected_page == "📈 Analytics":
        render_analytics_page(orchestrator)
    elif selected_page == "🤖 Assistant":
        render_assistant_page(orchestrator)
    elif selected_page == "📋 Reports":
        render_reports_page(orchestrator)


if __name__ == "__main__":
    main()
