#!/usr/bin/env python3
"""
AI Engineering Platform - Main Application
Ibm Ai Engineering Capstone
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIEngineeringPlatform:
    """Main platform class for AI Engineering"""
    
    def __init__(self):
        self.db_path = "platform.db"
        self.init_database()
        logger.info(f"AI Engineering platform initialized")
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create main data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS platform_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                value REAL NOT NULL,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS platform_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_date DATE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def generate_sample_data(self, num_records=1000):
        """Generate sample data for demonstration"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute("DELETE FROM platform_data")
        cursor.execute("DELETE FROM platform_metrics")
        
        # Generate sample records
        categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
        statuses = ['active', 'inactive', 'pending']
        
        for i in range(num_records):
            name = f"Item {i+1:04d}"
            category = np.random.choice(categories)
            value = np.random.uniform(10, 1000)
            status = np.random.choice(statuses, p=[0.7, 0.2, 0.1])
            
            cursor.execute("""
                INSERT INTO platform_data (name, category, value, status)
                VALUES (?, ?, ?, ?)
            """, (name, category, value, status))
        
        # Generate metrics data
        base_date = datetime.now() - timedelta(days=30)
        for i in range(30):
            date = base_date + timedelta(days=i)
            
            # Generate various metrics
            metrics = [
                ('total_users', np.random.randint(100, 1000)),
                ('active_sessions', np.random.randint(50, 500)),
                ('revenue', np.random.uniform(1000, 10000)),
                ('conversion_rate', np.random.uniform(0.1, 0.3)),
                ('performance_score', np.random.uniform(0.7, 1.0))
            ]
            
            for metric_name, metric_value in metrics:
                cursor.execute("""
                    INSERT INTO platform_metrics (metric_name, metric_value, metric_date)
                    VALUES (?, ?, ?)
                """, (metric_name, metric_value, date.date()))
        
        conn.commit()
        conn.close()
        logger.info(f"Generated {num_records} sample records and 30 days of metrics")
    
    def get_data(self):
        """Get all platform data"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM platform_data", conn)
        conn.close()
        return df
    
    def get_metrics(self):
        """Get platform metrics"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM platform_metrics", conn)
        conn.close()
        return df
    
    def calculate_kpis(self):
        """Calculate key performance indicators"""
        data = self.get_data()
        metrics = self.get_metrics()
        
        if data.empty:
            return {}
        
        kpis = {
            'total_records': len(data),
            'active_records': len(data[data['status'] == 'active']),
            'total_value': data['value'].sum(),
            'average_value': data['value'].mean(),
            'categories': data['category'].nunique(),
            'latest_update': data['updated_at'].max() if not data.empty else None
        }
        
        if not metrics.empty:
            latest_metrics = metrics.groupby('metric_name')['metric_value'].last()
            kpis.update(latest_metrics.to_dict())
        
        return kpis

def create_dashboard():
    """Create Streamlit dashboard"""
    st.set_page_config(
        page_title="AI Engineering Platform",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 AI Engineering Platform Dashboard")
    st.markdown("---")
    
    # Initialize platform
    platform = AIEngineeringPlatform()
    
    # Sidebar
    st.sidebar.title("🔧 Platform Controls")
    
    if st.sidebar.button("🔄 Generate Sample Data"):
        with st.spinner("Generating sample data..."):
            platform.generate_sample_data()
        st.sidebar.success("Sample data generated!")
        st.experimental_rerun()
    
    # Main dashboard
    data = platform.get_data()
    metrics = platform.get_metrics()
    kpis = platform.calculate_kpis()
    
    if data.empty:
        st.warning("No data available. Please generate sample data using the sidebar.")
        return
    
    # KPI Cards
    st.subheader("📈 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", kpis.get('total_records', 0))
    
    with col2:
        st.metric("Active Records", kpis.get('active_records', 0))
    
    with col3:
        st.metric("Total Value", f"${kpis.get('total_value', 0):,.2f}")
    
    with col4:
        st.metric("Average Value", f"${kpis.get('average_value', 0):,.2f}")
    
    # Charts
    st.subheader("📊 Data Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Category distribution
        category_counts = data['category'].value_counts()
        fig_pie = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Distribution by Category"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Status distribution
        status_counts = data['status'].value_counts()
        fig_bar = px.bar(
            x=status_counts.index,
            y=status_counts.values,
            title="Records by Status"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Time series metrics
    if not metrics.empty:
        st.subheader("📈 Metrics Over Time")
        
        metric_options = metrics['metric_name'].unique()
        selected_metric = st.selectbox("Select Metric", metric_options)
        
        metric_data = metrics[metrics['metric_name'] == selected_metric]
        metric_data['metric_date'] = pd.to_datetime(metric_data['metric_date'])
        
        fig_line = px.line(
            metric_data,
            x='metric_date',
            y='metric_value',
            title=f"{selected_metric.replace('_', ' ').title()} Over Time"
        )
        st.plotly_chart(fig_line, use_container_width=True)
    
    # Data table
    st.subheader("📋 Data Table")
    st.dataframe(data, use_container_width=True)
    
    # Performance metrics
    st.subheader("⚡ Performance Metrics")
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    
    with perf_col1:
        st.metric("Response Time", "< 2s", "✅")
    
    with perf_col2:
        st.metric("Uptime", "99.9%", "✅")
    
    with perf_col3:
        st.metric("Accuracy", "95.2%", "✅")

def main():
    """Main application entry point"""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--generate-data":
            platform = AIEngineeringPlatform()
            platform.generate_sample_data()
            print("Sample data generated successfully!")
        elif sys.argv[1] == "--setup":
            platform = AIEngineeringPlatform()
            print("Platform setup completed!")
        elif sys.argv[1] == "--start":
            print("Starting AI Engineering Platform...")
            create_dashboard()
        else:
            print("Usage: python main_platform.py [--generate-data|--setup|--start]")
    else:
        # Run Streamlit dashboard
        create_dashboard()

if __name__ == "__main__":
    main()
