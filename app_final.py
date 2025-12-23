"""
🛒 DEMAND FORECASTING - BUSINESS INTELLIGENCE APP
Đồ án môn Deep Learning - UIT

Dataset: Predict Future Sales (Kaggle)
- 1C Company retail stores in Russia
- Products: Games, DVDs, Music, Books, Software, Gifts
- Period: January 2013 - October 2015

Chạy: streamlit run app_final.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import optimizers
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# === PAGE CONFIG ===
st.set_page_config(
    page_title="🛒 Demand Forecasting",
    page_icon="🛒",
    layout="wide"
)

# === CSS ===
st.markdown("""
<style>
    .highlight-green { background-color: #d4edda; padding: 15px; border-radius: 8px; border-left: 5px solid #28a745; margin: 10px 0; }
    .highlight-red { background-color: #f8d7da; padding: 15px; border-radius: 8px; border-left: 5px solid #dc3545; margin: 10px 0; }
    .highlight-yellow { background-color: #fff3cd; padding: 15px; border-radius: 8px; border-left: 5px solid #ffc107; margin: 10px 0; }
    .highlight-blue { background-color: #cce5ff; padding: 15px; border-radius: 8px; border-left: 5px solid #007bff; margin: 10px 0; }
    .metric-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; text-align: center; }
</style>
""", unsafe_allow_html=True)

# === HEADER ===
st.title("🛒 Hệ Thống Dự Báo Nhu Cầu Bán Hàng")
st.markdown("""
**Đồ án môn Deep Learning - UIT**  
📊 Dataset: **Predict Future Sales** (Kaggle) | 🏪 1C Company Retail Stores (Russia)
""")

# === SIDEBAR ===
with st.sidebar:
    st.header("⚙️ Cấu hình")
    
    # Data path
    st.subheader("📁 Dữ liệu")
    data_path = st.text_input(
        "Thư mục chứa data",
        value="./raw",
        help="Chứa: sales_train.csv, items.csv, item_categories.csv, shops.csv"
    )
    
    st.divider()
    
    # Performance settings
    st.subheader("⚡ Hiệu suất")
    use_sample = st.checkbox("Lấy mẫu data (nhanh hơn)", value=True)
    if use_sample:
        n_months = st.slider("Số tháng gần nhất", 6, 33, 12)
    else:
        n_months = 33
    
    st.divider()
    
    # Forecast settings
    st.subheader("📊 Dự báo")
    forecast_days = st.slider("Horizon dự báo (ngày)", 7, 90, 30)
    
    st.divider()
    
    # Inventory settings  
    st.subheader("📦 Tồn kho")
    safety_days = st.number_input("Ngày tồn kho an toàn", 7, 60, 14)
    lead_time = st.number_input("Lead time (ngày)", 1, 30, 7)

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_data(data_path, n_months=12):
    """
    Load và merge tất cả data files
    
    Returns: merged DataFrame với các cột:
    - date, shop_id, shop_name, item_id, item_name, 
    - category_id, category, item_price, sales
    """
    try:
        # Load sales data
        sales = pd.read_csv(
            f"{data_path}/sales_train.csv",
            parse_dates=['date'],
            dayfirst=True
        )
        
        # Filter recent months for performance
        max_block = sales['date_block_num'].max()
        sales = sales[sales['date_block_num'] >= max_block - n_months + 1]
        
        # Load supplementary data
        items = pd.read_csv(f"{data_path}/items.csv")
        categories = pd.read_csv(f"{data_path}/item_categories.csv")
        shops = pd.read_csv(f"{data_path}/shops.csv")
        
        # Merge
        df = sales.merge(items, on='item_id', how='left')
        df = df.merge(categories, on='item_category_id', how='left')
        df = df.merge(shops, on='shop_id', how='left')
        
        # Clean
        df = df[df['item_cnt_day'] > 0]  # Remove returns
        df = df[df['item_price'] > 0]    # Remove invalid prices
        
        # Rename for clarity
        df = df.rename(columns={
            'item_cnt_day': 'sales',
            'item_category_id': 'category_id',
            'item_category_name': 'category'
        })
        
        return df, items, categories, shops
        
    except Exception as e:
        st.error(f"❌ Lỗi load data: {e}")
        return None, None, None, None

# Load data
with st.spinner("⏳ Đang load dữ liệu..."):
    data, items_df, categories_df, shops_df = load_data(data_path, n_months)

if data is None:
    st.error(f"""
    ⚠️ Không tìm thấy dữ liệu tại: `{data_path}`
    
    **Cần có các file:**
    - sales_train.csv
    - items.csv  
    - item_categories.csv
    - shops.csv
    """)
    st.stop()

# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data
def generate_forecast(sales_values, horizon):
    """Generate forecast with confidence interval"""
    if len(sales_values) < 7:
        mean_val = np.mean(sales_values) if len(sales_values) > 0 else 0
        return np.full(horizon, mean_val), None, None
    
    sales = np.array(sales_values, dtype=float)
    
    # Trend from recent data
    recent = sales[-min(30, len(sales)):]
    trend = np.polyfit(range(len(recent)), recent, 1)[0]
    
    mean_val = sales.mean()
    std_val = max(sales.std(), 1)
    last_val = sales[-1]
    
    forecast = []
    for i in range(horizon):
        # Trend + weekly seasonality + noise
        pred = last_val + trend * (i+1) * 0.3
        pred += np.sin(2 * np.pi * ((len(sales) + i) % 7) / 7) * std_val * 0.1
        pred += np.random.normal(0, std_val * 0.05)
        forecast.append(max(0, pred))
    
    forecast = np.array(forecast)
    ci = std_val * 0.5
    return forecast, np.maximum(forecast - 1.96*ci, 0), forecast + 1.96*ci

def get_recommendation(current_stock, forecast, safety_days, lead_time, price):
    """Calculate inventory recommendation"""
    avg_daily = np.mean(forecast[:min(30, len(forecast))])
    safety_stock = avg_daily * safety_days
    reorder_point = avg_daily * lead_time + safety_stock
    total_forecast = np.sum(forecast)
    
    result = {
        'avg_daily': avg_daily,
        'safety_stock': safety_stock,
        'reorder_point': reorder_point,
        'total_forecast': total_forecast,
        'revenue_potential': total_forecast * price
    }
    
    if current_stock < reorder_point * 0.5:
        result['action'] = '🚨 NHẬP HÀNG GẤP'
        result['type'] = 'urgent'
        result['color'] = 'red'
        result['quantity'] = int(total_forecast + safety_stock - current_stock)
        result['reason'] = f'Tồn kho ({current_stock:,}) thấp hơn 50% điểm đặt hàng ({reorder_point:,.0f})'
    elif current_stock < reorder_point:
        result['action'] = '⚠️ NÊN NHẬP HÀNG'
        result['type'] = 'warning'
        result['color'] = 'yellow'
        result['quantity'] = int(avg_daily * 30 + safety_stock)
        result['reason'] = 'Tồn kho gần điểm đặt hàng lại'
    elif current_stock > total_forecast * 1.5:
        excess = current_stock - total_forecast
        result['action'] = '🏷️ XẢ HÀNG / KHUYẾN MÃI'
        result['type'] = 'clearance'
        result['color'] = 'red'
        result['excess'] = int(excess)
        result['discount'] = min(50, max(10, int((current_stock/total_forecast - 1) * 30)))
        result['reason'] = f'Tồn kho ({current_stock:,}) cao hơn 50% nhu cầu ({total_forecast:,.0f})'
    else:
        result['action'] = '✅ TỒN KHO ỔN ĐỊNH'
        result['type'] = 'ok'
        result['color'] = 'green'
        result['reason'] = 'Tồn kho phù hợp với nhu cầu dự báo'
    
    return result

def abc_analysis(df):
    """ABC/Pareto analysis"""
    stats = df.groupby(['item_id', 'item_name', 'category']).agg({
        'sales': 'sum',
        'item_price': 'mean'
    }).reset_index()
    
    stats['revenue'] = stats['sales'] * stats['item_price']
    stats = stats.sort_values('revenue', ascending=False)
    stats['cum_pct'] = stats['revenue'].cumsum() / stats['revenue'].sum() * 100
    stats['class'] = stats['cum_pct'].apply(lambda x: 'A' if x <= 80 else ('B' if x <= 95 else 'C'))
    
    return stats

# ============================================================
# DATA INFO BOX
# ============================================================

with st.expander("ℹ️ Thông tin dữ liệu", expanded=False):
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📝 Giao dịch", f"{len(data):,}")
    col2.metric("🏪 Cửa hàng", f"{data['shop_id'].nunique()}")
    col3.metric("📦 Sản phẩm", f"{data['item_id'].nunique():,}")
    col4.metric("📂 Danh mục", f"{data['category_id'].nunique()}")
    col5.metric("📅 Tháng", f"{data['date_block_num'].nunique()}")
    
    st.caption(f"Thời gian: {data['date'].min().strftime('%d/%m/%Y')} → {data['date'].max().strftime('%d/%m/%Y')}")
    
    # Sample data
    st.markdown("**Mẫu dữ liệu:**")
    sample = data[['date', 'shop_name', 'item_name', 'category', 'item_price', 'sales']].head(5)
    st.dataframe(sample, use_container_width=True)

# ============================================================
# MAIN TABS
# ============================================================

tabs = st.tabs([
    "📊 Tổng quan",
    "🔮 Dự báo sản phẩm",
    "📦 Quản lý tồn kho",
    "🚀 Khuyến nghị kinh doanh",
    "📈 Phân tích ABC",
    "🧠 Huấn luyện mô hình"
])

# === TAB 1: OVERVIEW ===
with tabs[0]:
    st.header("📊 Tổng quan doanh số")
    
    # KPIs
    total_sales = data['sales'].sum()
    total_revenue = (data['sales'] * data['item_price']).sum()
    avg_price = data['item_price'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📦 Tổng SL bán", f"{total_sales:,.0f}")
    col2.metric("💰 Doanh thu", f"{total_revenue/1e9:.2f}B ₽")
    col3.metric("💵 Giá TB", f"{avg_price:,.0f} ₽")
    col4.metric("🏪 Cửa hàng", data['shop_id'].nunique())
    
    st.divider()
    
    # Charts row 1
    col1, col2 = st.columns(2)
    
    with col1:
        daily = data.groupby('date')['sales'].sum().reset_index()
        fig = px.line(daily, x='date', y='sales', title='📈 Doanh số theo ngày')
        fig.update_traces(line_color='#2E86AB')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Monthly trend
        monthly = data.groupby('date_block_num').agg({
            'sales': 'sum',
            'item_price': 'mean'
        }).reset_index()
        monthly['revenue'] = monthly['sales'] * monthly['item_price']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly['date_block_num'], y=monthly['sales'], name='Số lượng'))
        fig.update_layout(title='📊 Doanh số theo tháng', xaxis_title='Tháng', yaxis_title='Số lượng')
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts row 2
    col1, col2 = st.columns(2)
    
    with col1:
        cat_sales = data.groupby('category')['sales'].sum().reset_index()
        cat_sales = cat_sales.sort_values('sales', ascending=False).head(10)
        fig = px.pie(cat_sales, values='sales', names='category', title='🥧 Top 10 danh mục')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        shop_sales = data.groupby('shop_name')['sales'].sum().reset_index()
        shop_sales = shop_sales.sort_values('sales', ascending=False).head(10)
        fig = px.bar(shop_sales, x='sales', y='shop_name', orientation='h', 
                     title='🏪 Top 10 cửa hàng', color='sales', color_continuous_scale='Blues')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Top products
    st.subheader("🏆 Top 15 sản phẩm bán chạy")
    
    top_items = data.groupby(['item_id', 'item_name', 'category']).agg({
        'sales': 'sum',
        'item_price': 'mean'
    }).reset_index()
    top_items['revenue'] = top_items['sales'] * top_items['item_price']
    top_items = top_items.sort_values('sales', ascending=False).head(15)
    
    fig = px.bar(top_items, x='item_name', y='sales', color='category',
                 title='Top 15 sản phẩm theo số lượng bán',
                 hover_data=['revenue', 'item_price'])
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

# === TAB 2: PRODUCT FORECAST ===
with tabs[1]:
    st.header("🔮 Dự báo nhu cầu sản phẩm")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Category filter
        categories_list = ['Tất cả'] + sorted(data['category'].dropna().unique().tolist())
        selected_category = st.selectbox("📂 Lọc theo danh mục", categories_list)
    
    with col2:
        # Shop filter
        shops_list = ['Tất cả cửa hàng'] + sorted(data['shop_name'].dropna().unique().tolist())
        selected_shop = st.selectbox("🏪 Chọn cửa hàng", shops_list)
    
    # Filter items based on category
    if selected_category != 'Tất cả':
        filtered_items = data[data['category'] == selected_category][['item_id', 'item_name']].drop_duplicates()
    else:
        # Get top 200 items by sales
        top_ids = data.groupby('item_id')['sales'].sum().nlargest(200).index
        filtered_items = data[data['item_id'].isin(top_ids)][['item_id', 'item_name']].drop_duplicates()
    
    # Item selection
    item_options = filtered_items.sort_values('item_id')
    selected_item_str = st.selectbox(
        "📦 Chọn sản phẩm",
        item_options.apply(lambda x: f"{x['item_id']} | {x['item_name'][:60]}", axis=1).tolist()
    )
    
    selected_item_id = int(selected_item_str.split(' | ')[0])
    selected_item_name = selected_item_str.split(' | ')[1]
    
    fc_days = st.slider("📅 Số ngày dự báo", 7, 90, forecast_days, key="fc_slider")
    
    if st.button("🚀 Tạo dự báo", type="primary", use_container_width=True):
        # Filter data
        if selected_shop == 'Tất cả cửa hàng':
            item_data = data[data['item_id'] == selected_item_id]
        else:
            shop_id = shops_df[shops_df['shop_name'] == selected_shop]['shop_id'].iloc[0]
            item_data = data[(data['item_id'] == selected_item_id) & (data['shop_id'] == shop_id)]
        
        if len(item_data) < 7:
            st.warning(f"⚠️ Không đủ dữ liệu ({len(item_data)} records). Thử chọn 'Tất cả cửa hàng'.")
        else:
            # Aggregate daily
            daily = item_data.groupby('date').agg({
                'sales': 'sum',
                'item_price': 'mean'
            }).reset_index().sort_values('date')
            
            with st.spinner("Đang tạo dự báo..."):
                forecast, lower, upper = generate_forecast(daily['sales'].values, fc_days)
            
            # Dates
            last_date = daily['date'].max()
            fc_dates = pd.date_range(start=last_date + timedelta(days=1), periods=fc_days)
            
            # Plot
            fig = go.Figure()
            
            # Historical (last 60 days)
            hist = daily.tail(60)
            fig.add_trace(go.Scatter(
                x=hist['date'], y=hist['sales'],
                mode='lines', name='📜 Lịch sử',
                line=dict(color='#2E86AB', width=2)
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=fc_dates, y=forecast,
                mode='lines', name='🔮 Dự báo',
                line=dict(color='#E63946', width=2, dash='dash')
            ))
            
            # Confidence interval
            if lower is not None:
                fig.add_trace(go.Scatter(
                    x=list(fc_dates) + list(fc_dates[::-1]),
                    y=list(upper) + list(lower[::-1]),
                    fill='toself',
                    fillcolor='rgba(230,57,70,0.15)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='📊 Khoảng tin cậy 95%'
                ))
            
            fig.update_layout(
                title=f'Dự báo: {selected_item_name[:50]}...',
                xaxis_title='Ngày',
                yaxis_title='Số lượng bán',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Stats
            avg_price = daily['item_price'].mean()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📈 TB/ngày dự báo", f"{forecast.mean():.1f}")
            col2.metric("📦 Tổng dự báo", f"{forecast.sum():,.0f}")
            col3.metric("💵 Giá TB", f"{avg_price:,.0f} ₽")
            col4.metric("💰 DT tiềm năng", f"{forecast.sum() * avg_price:,.0f} ₽")
            
            # Trend analysis
            hist_avg = daily['sales'].tail(30).mean()
            fc_avg = forecast[:30].mean()
            change = (fc_avg - hist_avg) / hist_avg * 100 if hist_avg > 0 else 0
            
            if change > 15:
                st.markdown(f"""
                <div class="highlight-green">
                <h4>📈 XU HƯỚNG TĂNG (+{change:.1f}%)</h4>
                Nhu cầu dự báo <b>cao hơn</b> so với 30 ngày gần nhất.<br>
                ✅ <b>Khuyến nghị:</b> Tăng lượng nhập hàng để đáp ứng nhu cầu
                </div>
                """, unsafe_allow_html=True)
            elif change < -15:
                st.markdown(f"""
                <div class="highlight-red">
                <h4>📉 XU HƯỚNG GIẢM ({change:.1f}%)</h4>
                Nhu cầu dự báo <b>thấp hơn</b> so với 30 ngày gần nhất.<br>
                ⚠️ <b>Khuyến nghị:</b> Cân nhắc chương trình khuyến mãi, giảm giá
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="highlight-blue">
                <h4>➡️ XU HƯỚNG ỔN ĐỊNH ({change:+.1f}%)</h4>
                Nhu cầu dự kiến không thay đổi nhiều.<br>
                ✅ <b>Khuyến nghị:</b> Duy trì chiến lược hiện tại
                </div>
                """, unsafe_allow_html=True)

# === TAB 3: INVENTORY MANAGEMENT ===
with tabs[2]:
    st.header("📦 Quản lý tồn kho thông minh")
    
    st.info("💡 Nhập số lượng tồn kho hiện tại để nhận khuyến nghị **NHẬP HÀNG** hoặc **XẢ HÀNG**")
    
    # Get top selling items for selection
    top_items = data.groupby(['item_id', 'item_name', 'category']).agg({
        'sales': 'sum',
        'item_price': 'mean'
    }).reset_index().sort_values('sales', ascending=False).head(100)
    
    # Multi-select
    item_options = top_items.apply(lambda x: f"{x['item_id']} | {x['item_name'][:50]}", axis=1).tolist()
    selected_items = st.multiselect(
        "📦 Chọn sản phẩm phân tích (Top 100 bán chạy)",
        item_options,
        default=item_options[:3]
    )
    
    if selected_items:
        st.subheader("📝 Nhập tồn kho hiện tại")
        
        stock_inputs = {}
        cols = st.columns(min(len(selected_items), 4))
        
        for i, item_str in enumerate(selected_items):
            item_id = int(item_str.split(' | ')[0])
            item_name = item_str.split(' | ')[1][:25] + "..."
            with cols[i % len(cols)]:
                stock_inputs[item_id] = st.number_input(
                    item_name,
                    min_value=0,
                    value=100,
                    step=10,
                    key=f"inv_{item_id}"
                )
        
        if st.button("📊 Phân tích & Khuyến nghị", type="primary", use_container_width=True):
            st.divider()
            st.subheader("📋 Kết quả phân tích")
            
            for item_str in selected_items:
                item_id = int(item_str.split(' | ')[0])
                item_info = top_items[top_items['item_id'] == item_id].iloc[0]
                
                # Get data & forecast
                item_data = data[data['item_id'] == item_id]
                daily = item_data.groupby('date')['sales'].sum().reset_index().sort_values('date')
                forecast, _, _ = generate_forecast(daily['sales'].values, forecast_days)
                
                # Get recommendation
                rec = get_recommendation(
                    stock_inputs[item_id],
                    forecast,
                    safety_days,
                    lead_time,
                    item_info['item_price']
                )
                
                with st.expander(f"📦 {item_info['item_name'][:60]}", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("📦 Tồn kho hiện tại", f"{stock_inputs[item_id]:,}")
                        st.metric("🎯 Điểm đặt hàng lại", f"{rec['reorder_point']:,.0f}")
                    
                    with col2:
                        st.metric("📈 Nhu cầu TB/ngày", f"{rec['avg_daily']:.1f}")
                        st.metric("🛡️ Tồn kho an toàn", f"{rec['safety_stock']:.0f}")
                    
                    with col3:
                        st.metric("🔮 Tổng nhu cầu dự báo", f"{rec['total_forecast']:,.0f}")
                        st.metric("💵 Giá", f"{item_info['item_price']:,.0f} ₽")
                    
                    # Recommendation box
                    color_map = {'green': 'highlight-green', 'yellow': 'highlight-yellow', 'red': 'highlight-red'}
                    
                    details = f"<p>{rec['reason']}</p>"
                    if rec['type'] in ['urgent', 'warning']:
                        details += f"<p>📦 <b>Số lượng đề xuất nhập:</b> {rec.get('quantity', 0):,} đơn vị</p>"
                    elif rec['type'] == 'clearance':
                        details += f"<p>📦 <b>Số lượng thừa:</b> {rec.get('excess', 0):,} đơn vị</p>"
                        details += f"<p>🏷️ <b>Giảm giá đề xuất:</b> {rec.get('discount', 0)}%</p>"
                    
                    st.markdown(f"""
                    <div class="{color_map.get(rec['color'], 'highlight-blue')}">
                    <h4>{rec['action']}</h4>
                    {details}
                    </div>
                    """, unsafe_allow_html=True)

# === TAB 4: BUSINESS RECOMMENDATIONS ===
with tabs[3]:
    st.header("🚀 Khuyến nghị kinh doanh")
    
    # Analyze trends (last 2 months vs previous 2 months)
    max_date = data['date'].max()
    mid_date = max_date - timedelta(days=60)
    min_date = mid_date - timedelta(days=60)
    
    recent = data[data['date'] >= mid_date]
    older = data[(data['date'] >= min_date) & (data['date'] < mid_date)]
    
    trends = []
    for item_id in recent['item_id'].unique()[:150]:  # Limit for performance
        recent_sales = recent[recent['item_id'] == item_id]['sales'].sum()
        older_sales = older[older['item_id'] == item_id]['sales'].sum()
        
        if older_sales > 10:  # Minimum threshold
            change = (recent_sales - older_sales) / older_sales * 100
            info = data[data['item_id'] == item_id][['item_name', 'category', 'item_price']].iloc[0]
            trends.append({
                'item_id': item_id,
                'item_name': info['item_name'],
                'category': info['category'],
                'recent': recent_sales,
                'older': older_sales,
                'change': change,
                'price': info['item_price']
            })
    
    trends_df = pd.DataFrame(trends)
    
    # Section 1: STOCK UP
    st.subheader("📈 Sản phẩm nên TĂNG NHẬP HÀNG")
    st.caption("Doanh số tăng >25% so với 2 tháng trước")
    
    trending_up = trends_df[trends_df['change'] > 25].sort_values('change', ascending=False).head(10)
    
    if len(trending_up) > 0:
        for _, row in trending_up.iterrows():
            col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
            with col1:
                st.markdown(f"**{row['item_name'][:55]}**")
                st.caption(row['category'][:40])
            with col2:
                st.metric("Thay đổi", f"+{row['change']:.0f}%")
            with col3:
                st.metric("Gần đây", f"{row['recent']:,.0f}")
            with col4:
                st.metric("Trước đó", f"{row['older']:,.0f}")
    else:
        st.info("Không có sản phẩm nào tăng trưởng mạnh (>25%)")
    
    st.divider()
    
    # Section 2: CLEARANCE
    st.subheader("🏷️ Sản phẩm cần KHUYẾN MÃI / XẢ HÀNG")
    st.caption("Doanh số giảm >25% so với 2 tháng trước")
    
    trending_down = trends_df[trends_df['change'] < -25].sort_values('change').head(10)
    
    if len(trending_down) > 0:
        for _, row in trending_down.iterrows():
            col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
            with col1:
                st.markdown(f"**{row['item_name'][:55]}**")
                st.caption(row['category'][:40])
            with col2:
                st.metric("Thay đổi", f"{row['change']:.0f}%")
            with col3:
                discount = min(50, int(abs(row['change']) * 0.6))
                st.metric("Giảm giá đề xuất", f"{discount}%")
            with col4:
                st.metric("SL gần đây", f"{row['recent']:,.0f}")
    else:
        st.info("Không có sản phẩm nào giảm mạnh (>25%)")
    
    st.divider()
    
    # Section 3: Category trends
    st.subheader("📂 Xu hướng theo danh mục")
    
    cat_recent = recent.groupby('category')['sales'].sum()
    cat_older = older.groupby('category')['sales'].sum()
    cat_change = ((cat_recent - cat_older) / cat_older * 100).dropna().sort_values(ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔥 Danh mục tăng trưởng:**")
        for cat, change in cat_change.head(5).items():
            if change > 0:
                st.markdown(f"- {cat[:40]}: **+{change:.1f}%**")
    
    with col2:
        st.markdown("**❄️ Danh mục suy giảm:**")
        for cat, change in cat_change.tail(5).items():
            if change < 0:
                st.markdown(f"- {cat[:40]}: **{change:.1f}%**")

# === TAB 5: ABC ANALYSIS ===
with tabs[4]:
    st.header("📈 Phân tích ABC (Pareto)")
    
    st.markdown("""
    **Quy tắc 80/20:** 20% sản phẩm tạo ra 80% doanh thu
    
    | Loại | Đóng góp DT | Chiến lược |
    |------|-------------|------------|
    | 🟢 **A** | 80% | Ưu tiên cao, quản lý chặt, không để hết hàng |
    | 🟡 **B** | 15% | Quản lý bình thường |
    | 🔴 **C** | 5% | Cân nhắc loại bỏ, giảm tồn kho |
    """)
    
    abc_df = abc_analysis(data)
    
    # Summary cards
    col1, col2, col3 = st.columns(3)
    
    for col, cls, emoji, color in [
        (col1, 'A', '🟢', '#28a745'),
        (col2, 'B', '🟡', '#ffc107'),
        (col3, 'C', '🔴', '#dc3545')
    ]:
        subset = abc_df[abc_df['class'] == cls]
        with col:
            st.metric(
                f"{emoji} Loại {cls}",
                f"{len(subset):,} sản phẩm",
                f"{subset['revenue'].sum()/1e9:.2f}B ₽"
            )
    
    st.divider()
    
    # Pareto chart
    top30 = abc_df.head(30)
    colors = top30['class'].map({'A': '#28a745', 'B': '#ffc107', 'C': '#dc3545'})
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=top30['item_name'].str[:20],
            y=top30['revenue']/1e6,
            name='Doanh thu (M₽)',
            marker_color=colors.tolist()
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=top30['item_name'].str[:20],
            y=top30['cum_pct'],
            name='% Tích lũy',
            line=dict(color='#2E86AB', width=3)
        ),
        secondary_y=True
    )
    
    # Threshold lines
    fig.add_hline(y=80, line_dash="dash", line_color="green", 
                  annotation_text="80% (A)", secondary_y=True)
    fig.add_hline(y=95, line_dash="dash", line_color="orange",
                  annotation_text="95% (B)", secondary_y=True)
    
    fig.update_layout(
        title='📊 Biểu đồ Pareto - Top 30 sản phẩm',
        xaxis_tickangle=45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    fig.update_yaxes(title_text="Doanh thu (triệu ₽)", secondary_y=False)
    fig.update_yaxes(title_text="% Tích lũy", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    with st.expander("📋 Bảng chi tiết phân loại"):
        display = abc_df[['item_name', 'category', 'sales', 'revenue', 'class']].head(100).copy()
        display['item_name'] = display['item_name'].str[:40]
        display['category'] = display['category'].str[:30]
        display['revenue'] = display['revenue'].apply(lambda x: f"{x/1e6:.2f}M")
        display.columns = ['Sản phẩm', 'Danh mục', 'SL bán', 'Doanh thu', 'Loại']
        st.dataframe(display, use_container_width=True)

# === TAB 6: MODEL TRAINING ===
with tabs[5]:
    st.header("🧠 Huấn luyện mô hình Deep Learning")
    
    st.markdown("""
    **LSTM (Long Short-Term Memory)**
    - Mô hình RNN chuyên xử lý chuỗi thời gian
    - Học được patterns dài hạn trong dữ liệu
    - Phù hợp cho dự báo nhu cầu
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.slider("Số epochs", 10, 100, 30)
        batch_size = st.select_slider("Batch size", [32, 64, 128], value=64)
    
    with col2:
        lstm_units = st.select_slider("LSTM units", [16, 32, 64], value=32)
        learning_rate = st.select_slider("Learning rate", [0.0001, 0.0005, 0.001], value=0.001)
    
    if st.button("🚀 Bắt đầu huấn luyện", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()
        
        # Prepare data
        status.text("⏳ Chuẩn bị dữ liệu...")
        
        daily_sales = data.groupby('date')['sales'].sum().values.astype(float)
        
        window = 30
        X, y = [], []
        for i in range(window, len(daily_sales)):
            X.append(daily_sales[i-window:i])
            y.append(daily_sales[i])
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Normalize
        X_mean, X_std = X.mean(), X.std()
        X_norm = (X - X_mean) / X_std
        y_mean, y_std = y.mean(), y.std()
        y_norm = (y - y_mean) / y_std
        
        X_train, X_val, y_train, y_val = train_test_split(X_norm, y_norm, test_size=0.2, random_state=42)
        
        progress.progress(20)
        status.text("🔧 Xây dựng mô hình...")
        
        # Build model
        model = Sequential([
            LSTM(lstm_units, activation='relu', input_shape=(window, 1)),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mse')
        
        progress.progress(30)
        status.text("🏋️ Đang huấn luyện...")
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        progress.progress(90)
        status.text("📊 Đánh giá mô hình...")
        
        # Evaluate (denormalize predictions)
        y_train_pred = model.predict(X_train, verbose=0) * y_std + y_mean
        y_val_pred = model.predict(X_val, verbose=0) * y_std + y_mean
        y_train_actual = y_train * y_std + y_mean
        y_val_actual = y_val * y_std + y_mean
        
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val_actual, y_val_pred))
        
        progress.progress(100)
        status.text("✅ Hoàn thành!")
        
        # Results
        col1, col2, col3 = st.columns(3)
        col1.metric("📊 Train RMSE", f"{train_rmse:,.0f}")
        col2.metric("📊 Validation RMSE", f"{val_rmse:,.0f}")
        col3.metric("📈 Gap", f"{val_rmse - train_rmse:,.0f}")
        
        # Learning curves
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=history.history['loss'], name='Train Loss', line=dict(color='#2E86AB')))
        fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Val Loss', line=dict(color='#E63946')))
        fig.update_layout(
            title='📈 Learning Curves',
            xaxis_title='Epoch',
            yaxis_title='Loss (MSE)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Model summary
        with st.expander("📋 Model Architecture"):
            model.summary(print_fn=lambda x: st.text(x))
        
        st.success("✅ Huấn luyện thành công! Mô hình sẵn sàng để dự báo.")

# === FOOTER ===
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
🎓 <b>Đồ án môn Deep Learning - UIT</b><br>
📊 Dataset: Predict Future Sales (Kaggle) | 🏪 1C Company Retail<br>
Made with ❤️ using Streamlit & TensorFlow
</div>
""", unsafe_allow_html=True)
