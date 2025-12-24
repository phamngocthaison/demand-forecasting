"""
🛒 DEMAND FORECASTING - ML PREDICTION APP
Đồ án môn Deep Learning - UIT

Dataset: Predict Future Sales (Kaggle)
- 1C Company retail stores in Russia
- Products: Games, DVDs, Music, Books, Software, Gifts
- Period: January 2013 - October 2015

✅ Sử dụng model LSTM/CNN đã train từ notebook
✅ Fallback sang rule-based nếu không có model

Chạy: streamlit run app_final.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import optimizers
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# === PAGE CONFIG ===
st.set_page_config(
    page_title="🛒 Demand Forecasting ML",
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
    .model-status { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 12px 20px; border-radius: 10px; color: white; margin: 10px 0; }
    .model-status-warning { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 12px 20px; border-radius: 10px; color: white; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 🤖 MODEL LOADING - Load model đã train từ notebook
# ============================================================

@st.cache_resource
def load_trained_model(model_path):
    """
    Load model đã train từ file .h5
    
    Returns: model hoặc None nếu không tìm thấy
    """
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            return model
        except Exception as e:
            st.warning(f"⚠️ Không thể load model: {e}")
            return None
    return None


@st.cache_resource
def load_all_models(model_dir):
    """
    Load tất cả models có trong thư mục
    
    Returns: dict {model_name: model}
    """
    models = {}
    config = None
    
    # Load config nếu có
    config_path = f"{model_dir}/config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Các model files cần tìm
    model_files = {
        'LSTM': 'lstm_model.h5',
        'CNN': 'cnn_model.h5',
        'MLP': 'mlp_model.h5',
        'CNN-LSTM': 'cnn_lstm_model.h5'
    }
    
    for name, filename in model_files.items():
        path = f"{model_dir}/{filename}"
        if os.path.exists(path):
            try:
                models[name] = load_model(path)
            except Exception as e:
                pass  # Skip failed models
    
    return models, config


# ============================================================
# 🔮 FORECASTING FUNCTIONS - Dự báo bằng ML Model
# ============================================================

def predict_with_lstm(model, sales_history, horizon, window_size=30):
    """
    🤖 DỰ BÁO BẰNG LSTM MODEL
    
    Input shape: (samples, timesteps, features) = (1, 30, 1)
    
    Quy trình:
    1. Lấy `window_size` ngày gần nhất
    2. Reshape thành (1, window_size, 1)
    3. Predict → slide window → lặp lại cho đến horizon
    """
    sales = np.array(sales_history, dtype=float)
    
    # Pad nếu không đủ data
    if len(sales) < window_size:
        padding = np.full(window_size - len(sales), sales.mean())
        sales = np.concatenate([padding, sales])
    
    # Lấy window cuối cùng
    current_window = sales[-window_size:].copy()
    
    forecast = []
    for _ in range(horizon):
        # Reshape cho LSTM: (1, window_size, 1)
        X = current_window.reshape(1, window_size, 1)
        
        # Predict
        pred = model.predict(X, verbose=0)[0, 0]
        pred = max(0, pred)  # Không cho phép giá trị âm
        forecast.append(pred)
        
        # Slide window: bỏ phần tử đầu, thêm prediction vào cuối
        current_window = np.append(current_window[1:], pred)
    
    return np.array(forecast)


def predict_with_cnn(model, sales_history, horizon, window_size=30):
    """
    🤖 DỰ BÁO BẰNG CNN MODEL
    
    Input shape: (samples, timesteps, features) = (1, 30, 1)
    """
    sales = np.array(sales_history, dtype=float)
    
    if len(sales) < window_size:
        padding = np.full(window_size - len(sales), sales.mean())
        sales = np.concatenate([padding, sales])
    
    current_window = sales[-window_size:].copy()
    
    forecast = []
    for _ in range(horizon):
        X = current_window.reshape(1, window_size, 1)
        pred = model.predict(X, verbose=0)[0, 0]
        pred = max(0, pred)
        forecast.append(pred)
        current_window = np.append(current_window[1:], pred)
    
    return np.array(forecast)


def predict_with_mlp(model, sales_history, horizon, window_size=30):
    """
    🤖 DỰ BÁO BẰNG MLP MODEL
    
    Input shape: (samples, features) = (1, 30)
    """
    sales = np.array(sales_history, dtype=float)
    
    if len(sales) < window_size:
        padding = np.full(window_size - len(sales), sales.mean())
        sales = np.concatenate([padding, sales])
    
    current_window = sales[-window_size:].copy()
    
    forecast = []
    for _ in range(horizon):
        # MLP nhận input 2D, không cần chiều features
        X = current_window.reshape(1, window_size)
        pred = model.predict(X, verbose=0)[0, 0]
        pred = max(0, pred)
        forecast.append(pred)
        current_window = np.append(current_window[1:], pred)
    
    return np.array(forecast)


def predict_with_cnn_lstm(model, sales_history, horizon, window_size=30):
    """
    🤖 DỰ BÁO BẰNG CNN-LSTM MODEL
    
    Input shape: (samples, subsequences, timesteps, features) = (1, 2, 15, 1)
    """
    sales = np.array(sales_history, dtype=float)
    
    if len(sales) < window_size:
        padding = np.full(window_size - len(sales), sales.mean())
        sales = np.concatenate([padding, sales])
    
    current_window = sales[-window_size:].copy()
    
    forecast = []
    for _ in range(horizon):
        # CNN-LSTM cần reshape thành (1, 2, 15, 1) - chia thành 2 subsequences
        X = current_window.reshape(1, 2, 15, 1)
        pred = model.predict(X, verbose=0)[0, 0]
        pred = max(0, pred)
        forecast.append(pred)
        current_window = np.append(current_window[1:], pred)
    
    return np.array(forecast)


def forecast_rule_based(sales_history, horizon):
    """
    📏 FALLBACK: Dự báo bằng rule-based khi không có model
    
    Phương pháp: Linear trend + Weekly seasonality + Noise
    """
    sales = np.array(sales_history, dtype=float)
    
    if len(sales) < 7:
        mean_val = np.mean(sales) if len(sales) > 0 else 0
        return np.full(horizon, mean_val)
    
    # Trend từ 30 ngày gần nhất
    recent = sales[-min(30, len(sales)):]
    trend = np.polyfit(range(len(recent)), recent, 1)[0]
    
    std_val = max(sales.std(), 1)
    last_val = sales[-1]
    
    forecast = []
    for i in range(horizon):
        # Trend + weekly seasonality
        pred = last_val + trend * (i+1) * 0.3
        pred += np.sin(2 * np.pi * ((len(sales) + i) % 7) / 7) * std_val * 0.1
        forecast.append(max(0, pred))
    
    return np.array(forecast)


def generate_forecast(sales_history, horizon, model=None, model_type='LSTM', window_size=30):
    """
    🔮 HÀM DỰ BÁO CHÍNH
    
    - Nếu có model: sử dụng ML model để predict
    - Nếu không có model: fallback sang rule-based
    
    Returns: (forecast_array, method_used)
    """
    if model is None:
        forecast = forecast_rule_based(sales_history, horizon)
        return forecast, 'Rule-based'
    
    try:
        if model_type == 'LSTM':
            forecast = predict_with_lstm(model, sales_history, horizon, window_size)
        elif model_type == 'CNN':
            forecast = predict_with_cnn(model, sales_history, horizon, window_size)
        elif model_type == 'MLP':
            forecast = predict_with_mlp(model, sales_history, horizon, window_size)
        elif model_type == 'CNN-LSTM':
            forecast = predict_with_cnn_lstm(model, sales_history, horizon, window_size)
        else:
            forecast = forecast_rule_based(sales_history, horizon)
            return forecast, 'Rule-based'
        
        return forecast, model_type
    
    except Exception as e:
        # Fallback nếu có lỗi
        st.warning(f"⚠️ Lỗi predict với {model_type}: {e}. Chuyển sang rule-based.")
        forecast = forecast_rule_based(sales_history, horizon)
        return forecast, 'Rule-based'


def calculate_confidence_interval(forecast, sales_history):
    """Tính khoảng tin cậy 95%"""
    std_val = np.std(sales_history) if len(sales_history) > 1 else 1
    ci = std_val * 0.5
    lower = np.maximum(forecast - 1.96 * ci, 0)
    upper = forecast + 1.96 * ci
    return lower, upper


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_data(data_path, n_months=12):
    """Load và merge data từ Kaggle Predict Future Sales"""
    try:
        # Load sales data
        sales = pd.read_csv(
            f"{data_path}/sales_train.csv",
            parse_dates=['date'],
            dayfirst=True
        )
        
        # Filter recent months
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
        df = df[df['item_cnt_day'] > 0]
        df = df[df['item_price'] > 0]
        
        df = df.rename(columns={
            'item_cnt_day': 'sales',
            'item_category_id': 'category_id',
            'item_category_name': 'category'
        })
        
        return df, items, categories, shops
        
    except Exception as e:
        st.error(f"❌ Lỗi load data: {e}")
        return None, None, None, None


def get_recommendation(current_stock, forecast, safety_days, lead_time, price):
    """Tính toán khuyến nghị tồn kho"""
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
        result['action'] = '🏷️ XẢ HÀNG / KHUYẾN MÃI'
        result['type'] = 'clearance'
        result['color'] = 'red'
        result['excess'] = int(current_stock - total_forecast)
        result['discount'] = min(50, max(10, int((current_stock/total_forecast - 1) * 30)))
        result['reason'] = f'Tồn kho ({current_stock:,}) cao hơn 50% nhu cầu ({total_forecast:,.0f})'
    else:
        result['action'] = '✅ TỒN KHO ỔN ĐỊNH'
        result['type'] = 'ok'
        result['color'] = 'green'
        result['reason'] = 'Tồn kho phù hợp với nhu cầu dự báo'
    
    return result


def abc_analysis(df):
    """Phân tích ABC/Pareto"""
    stats = df.groupby(['item_id', 'item_name', 'category']).agg({
        'sales': 'sum',
        'item_price': 'mean'
    }).reset_index()
    
    stats['revenue'] = stats['sales'] * stats['item_price']
    stats = stats.sort_values('revenue', ascending=False)
    stats['cum_pct'] = stats['revenue'].cumsum() / stats['revenue'].sum() * 100
    stats['class'] = stats['cum_pct'].apply(lambda x: 'A' if x <= 80 else ('B' if x <= 95 else 'C'))
    
    return stats


# === HEADER ===
st.title("🛒 Hệ Thống Dự Báo Nhu Cầu Bán Hàng")
st.markdown("""
**Đồ án môn Deep Learning - UIT** | 🤖 Sử dụng ML Model (LSTM/CNN) đã train từ notebook
""")

# === SIDEBAR ===
with st.sidebar:
    st.header("⚙️ Cấu hình")
    
    # Data path
    st.subheader("📁 Dữ liệu")
    data_path = st.text_input("Thư mục data", value="./raw")
    
    st.divider()
    
    # Model settings
    st.subheader("🤖 ML Model")
    model_dir = st.text_input("Thư mục models", value="./models")
    
    # Load available models
    available_models, model_config = load_all_models(model_dir)
    
    if available_models:
        model_names = list(available_models.keys())
        selected_model_type = st.selectbox("Chọn model", model_names, index=0)
        selected_model = available_models[selected_model_type]
    else:
        selected_model_type = None
        selected_model = None
        st.warning("⚠️ Không tìm thấy model")
    
    window_size = st.number_input("Window size", 14, 60, 30, 
                                   help="Số ngày lịch sử để predict (phải khớp với model đã train)")
    
    st.divider()
    
    # Performance settings
    st.subheader("⚡ Hiệu suất")
    n_months = st.slider("Số tháng data", 6, 33, 12)
    
    st.divider()
    
    # Forecast settings
    st.subheader("📊 Dự báo")
    forecast_days = st.slider("Horizon (ngày)", 7, 90, 30)
    
    st.divider()
    
    # Inventory settings  
    st.subheader("📦 Tồn kho")
    safety_days = st.number_input("Ngày an toàn", 7, 60, 14)
    lead_time = st.number_input("Lead time", 1, 30, 7)

# === LOAD DATA ===
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

# === MODEL STATUS DISPLAY ===
col1, col2 = st.columns(2)

with col1:
    st.success(f"✅ **Data loaded:** {len(data):,} records | {data['item_id'].nunique():,} sản phẩm")

with col2:
    if selected_model is not None:
        st.markdown(f"""
        <div class="model-status">
        🤖 <b>Model: {selected_model_type}</b> | 📁 {model_dir}/{selected_model_type.lower()}_model.h5
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="model-status-warning">
        📏 <b>Rule-based</b> | Không tìm thấy model trong {model_dir}
        </div>
        """, unsafe_allow_html=True)

# Show model performance if available
if model_config and 'results' in model_config and selected_model_type:
    results = model_config['results']
    if selected_model_type in results:
        r = results[selected_model_type]
        st.caption(f"📊 **Model Performance:** Train RMSE = {r['train_rmse']:.2f} | Val RMSE = {r['val_rmse']:.2f}")

# ============================================================
# MAIN TABS
# ============================================================

tabs = st.tabs([
    "📊 Tổng quan",
    "🔮 Dự báo sản phẩm",
    "📦 Quản lý tồn kho",
    "🚀 Khuyến nghị",
    "📈 Phân tích ABC",
    "🧠 Model Info"
])

# === TAB 1: OVERVIEW ===
with tabs[0]:
    st.header("📊 Tổng quan doanh số")
    
    total_sales = data['sales'].sum()
    total_revenue = (data['sales'] * data['item_price']).sum()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📦 Tổng SL", f"{total_sales:,.0f}")
    col2.metric("💰 Doanh thu", f"{total_revenue/1e9:.2f}B ₽")
    col3.metric("📦 Sản phẩm", f"{data['item_id'].nunique():,}")
    col4.metric("🏪 Cửa hàng", data['shop_id'].nunique())
    
    col1, col2 = st.columns(2)
    
    with col1:
        daily = data.groupby('date')['sales'].sum().reset_index()
        fig = px.line(daily, x='date', y='sales', title='📈 Doanh số theo ngày')
        fig.update_traces(line_color='#2E86AB')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        cat_sales = data.groupby('category')['sales'].sum().nlargest(10).reset_index()
        fig = px.pie(cat_sales, values='sales', names='category', title='🥧 Top 10 danh mục')
        st.plotly_chart(fig, use_container_width=True)
    
    # Top products
    st.subheader("🏆 Top 15 sản phẩm bán chạy")
    top_items = data.groupby(['item_id', 'item_name', 'category']).agg({
        'sales': 'sum', 'item_price': 'mean'
    }).reset_index().nlargest(15, 'sales')
    
    fig = px.bar(top_items, x='item_name', y='sales', color='category',
                 hover_data=['item_price'])
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

# === TAB 2: PRODUCT FORECAST ===
with tabs[1]:
    st.header("🔮 Dự báo nhu cầu sản phẩm")
    
    # Model status
    if selected_model is not None:
        st.info(f"🤖 **Đang sử dụng: {selected_model_type} Model** (đã train từ notebook)")
    else:
        st.warning("📏 **Đang sử dụng: Rule-based** (chưa có model - hãy train trong notebook)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        categories_list = ['Tất cả'] + sorted(data['category'].dropna().unique().tolist())
        selected_category = st.selectbox("📂 Danh mục", categories_list)
    
    with col2:
        fc_days = st.slider("📅 Ngày dự báo", 7, 90, forecast_days, key="fc_slider")
    
    # Filter items
    if selected_category != 'Tất cả':
        filtered_items = data[data['category'] == selected_category][['item_id', 'item_name']].drop_duplicates()
    else:
        top_ids = data.groupby('item_id')['sales'].sum().nlargest(200).index
        filtered_items = data[data['item_id'].isin(top_ids)][['item_id', 'item_name']].drop_duplicates()
    
    item_options = filtered_items.sort_values('item_id')
    selected_item_str = st.selectbox(
        "📦 Sản phẩm",
        item_options.apply(lambda x: f"{x['item_id']} | {x['item_name'][:60]}", axis=1).tolist()
    )
    
    selected_item_id = int(selected_item_str.split(' | ')[0])
    selected_item_name = selected_item_str.split(' | ')[1]
    
    if st.button("🚀 Tạo dự báo", type="primary", use_container_width=True):
        item_data = data[data['item_id'] == selected_item_id]
        daily = item_data.groupby('date').agg({
            'sales': 'sum', 'item_price': 'mean'
        }).reset_index().sort_values('date')
        
        if len(daily) < 7:
            st.warning(f"⚠️ Không đủ dữ liệu ({len(daily)} ngày)")
        else:
            with st.spinner("🔮 Đang dự báo với ML model..."):
                # ========================================
                # 🤖 GỌI HÀM DỰ BÁO VỚI ML MODEL
                # ========================================
                forecast, method_used = generate_forecast(
                    sales_history=daily['sales'].values,
                    horizon=fc_days,
                    model=selected_model,
                    model_type=selected_model_type if selected_model else 'LSTM',
                    window_size=window_size
                )
                
                lower, upper = calculate_confidence_interval(forecast, daily['sales'].values)
            
            # Show method used
            if method_used != 'Rule-based':
                st.success(f"✅ Dự báo thành công bằng **{method_used} Model**")
            else:
                st.info(f"ℹ️ Dự báo bằng **Rule-based** (không có model)")
            
            # Plot
            last_date = daily['date'].max()
            fc_dates = pd.date_range(start=last_date + timedelta(days=1), periods=fc_days)
            
            fig = go.Figure()
            
            # Historical
            hist = daily.tail(60)
            fig.add_trace(go.Scatter(
                x=hist['date'], y=hist['sales'],
                mode='lines', name='📜 Lịch sử',
                line=dict(color='#2E86AB', width=2)
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=fc_dates, y=forecast,
                mode='lines', name=f'🔮 Dự báo ({method_used})',
                line=dict(color='#E63946', width=2, dash='dash')
            ))
            
            # CI
            fig.add_trace(go.Scatter(
                x=list(fc_dates) + list(fc_dates[::-1]),
                y=list(upper) + list(lower[::-1]),
                fill='toself', fillcolor='rgba(230,57,70,0.15)',
                line=dict(color='rgba(0,0,0,0)'), name='📊 95% CI'
            ))
            
            fig.update_layout(
                title=f'Dự báo: {selected_item_name[:50]}...',
                xaxis_title='Ngày', yaxis_title='Số lượng',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Stats
            avg_price = daily['item_price'].mean()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📈 TB/ngày", f"{forecast.mean():.1f}")
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
                Dự báo bằng <b>{method_used}</b> cho thấy nhu cầu tăng.<br>
                ✅ <b>Khuyến nghị:</b> Tăng lượng nhập hàng
                </div>
                """, unsafe_allow_html=True)
            elif change < -15:
                st.markdown(f"""
                <div class="highlight-red">
                <h4>📉 XU HƯỚNG GIẢM ({change:.1f}%)</h4>
                Dự báo bằng <b>{method_used}</b> cho thấy nhu cầu giảm.<br>
                ⚠️ <b>Khuyến nghị:</b> Cân nhắc khuyến mãi
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="highlight-blue">
                <h4>➡️ ỔN ĐỊNH ({change:+.1f}%)</h4>
                Dự báo bằng <b>{method_used}</b> cho thấy nhu cầu ổn định.<br>
                ✅ <b>Khuyến nghị:</b> Duy trì chiến lược hiện tại
                </div>
                """, unsafe_allow_html=True)

# === TAB 3: INVENTORY ===
with tabs[2]:
    st.header("📦 Quản lý tồn kho thông minh")
    
    if selected_model:
        st.info(f"🤖 Dự báo sử dụng **{selected_model_type} Model**")
    
    top_items = data.groupby(['item_id', 'item_name']).agg({
        'sales': 'sum', 'item_price': 'mean'
    }).reset_index().nlargest(50, 'sales')
    
    options = top_items.apply(lambda x: f"{x['item_id']} | {x['item_name'][:40]}", axis=1).tolist()
    selected = st.multiselect("Chọn sản phẩm (Top 50)", options, default=options[:3])
    
    if selected:
        st.subheader("📝 Nhập tồn kho")
        stocks = {}
        cols = st.columns(min(len(selected), 3))
        for i, item_str in enumerate(selected):
            item_id = int(item_str.split(' | ')[0])
            with cols[i % len(cols)]:
                stocks[item_id] = st.number_input(
                    item_str.split(' | ')[1][:20],
                    value=100, min_value=0, key=f"s_{item_id}"
                )
        
        if st.button("📊 Phân tích & Khuyến nghị", type="primary"):
            for item_str in selected:
                item_id = int(item_str.split(' | ')[0])
                info = top_items[top_items['item_id'] == item_id].iloc[0]
                
                item_data = data[data['item_id'] == item_id]
                daily = item_data.groupby('date')['sales'].sum().sort_values()
                
                # Dự báo bằng ML model
                forecast, method = generate_forecast(
                    daily.values, forecast_days,
                    selected_model, selected_model_type if selected_model else 'LSTM', window_size
                )
                
                rec = get_recommendation(stocks[item_id], forecast, safety_days, lead_time, info['item_price'])
                
                with st.expander(f"📦 {info['item_name'][:50]} ({method})", expanded=True):
                    col1, col2 = st.columns(2)
                    col1.metric("Tồn kho", f"{stocks[item_id]:,}")
                    col1.metric("Điểm đặt hàng", f"{rec['reorder_point']:,.0f}")
                    col2.metric("Nhu cầu TB/ngày", f"{rec['avg_daily']:.1f}")
                    col2.metric("Dự báo tổng", f"{rec['total_forecast']:,.0f}")
                    
                    color_map = {'green': 'highlight-green', 'yellow': 'highlight-yellow', 'red': 'highlight-red'}
                    
                    details = f"<p>{rec['reason']}</p>"
                    if rec['type'] in ['urgent', 'warning']:
                        details += f"<p>📦 <b>SL đề xuất nhập:</b> {rec.get('quantity', 0):,}</p>"
                    elif rec['type'] == 'clearance':
                        details += f"<p>📦 <b>SL thừa:</b> {rec.get('excess', 0):,}</p>"
                        details += f"<p>🏷️ <b>Giảm giá:</b> {rec.get('discount', 0)}%</p>"
                    
                    st.markdown(f"""
                    <div class="{color_map.get(rec['color'], 'highlight-blue')}">
                    <h4>{rec['action']}</h4>
                    {details}
                    </div>
                    """, unsafe_allow_html=True)

# === TAB 4: RECOMMENDATIONS ===
with tabs[3]:
    st.header("🚀 Khuyến nghị kinh doanh")
    
    max_date = data['date'].max()
    mid_date = max_date - timedelta(days=60)
    min_date = mid_date - timedelta(days=60)
    
    recent = data[data['date'] >= mid_date]
    older = data[(data['date'] >= min_date) & (data['date'] < mid_date)]
    
    trends = []
    for item_id in recent['item_id'].unique()[:100]:
        recent_sales = recent[recent['item_id'] == item_id]['sales'].sum()
        older_sales = older[older['item_id'] == item_id]['sales'].sum()
        
        if older_sales > 10:
            change = (recent_sales - older_sales) / older_sales * 100
            info = data[data['item_id'] == item_id][['item_name', 'category', 'item_price']].iloc[0]
            trends.append({
                'item_id': item_id, 'item_name': info['item_name'],
                'category': info['category'], 'recent': recent_sales,
                'older': older_sales, 'change': change, 'price': info['item_price']
            })
    
    trends_df = pd.DataFrame(trends)
    
    st.subheader("📈 Sản phẩm nên TĂNG NHẬP")
    trending_up = trends_df[trends_df['change'] > 25].sort_values('change', ascending=False).head(10)
    
    if len(trending_up) > 0:
        for _, row in trending_up.iterrows():
            col1, col2, col3 = st.columns([5, 1, 1])
            col1.markdown(f"**{row['item_name'][:50]}**")
            col2.metric("Thay đổi", f"+{row['change']:.0f}%")
            col3.metric("Gần đây", f"{row['recent']:,.0f}")
    else:
        st.info("Không có sản phẩm tăng mạnh")
    
    st.divider()
    
    st.subheader("🏷️ Sản phẩm cần KHUYẾN MÃI")
    trending_down = trends_df[trends_df['change'] < -25].sort_values('change').head(10)
    
    if len(trending_down) > 0:
        for _, row in trending_down.iterrows():
            col1, col2, col3 = st.columns([5, 1, 1])
            col1.markdown(f"**{row['item_name'][:50]}**")
            col2.metric("Thay đổi", f"{row['change']:.0f}%")
            discount = min(50, int(abs(row['change']) * 0.6))
            col3.metric("Giảm giá đề xuất", f"{discount}%")
    else:
        st.info("Không có sản phẩm giảm mạnh")

# === TAB 5: ABC ===
with tabs[4]:
    st.header("📈 Phân tích ABC (Pareto)")
    
    abc_df = abc_analysis(data)
    
    col1, col2, col3 = st.columns(3)
    for col, cls, emoji in [(col1, 'A', '🟢'), (col2, 'B', '🟡'), (col3, 'C', '🔴')]:
        subset = abc_df[abc_df['class'] == cls]
        col.metric(f"{emoji} Loại {cls}", f"{len(subset):,} SP", f"{subset['revenue'].sum()/1e9:.2f}B ₽")
    
    # Pareto chart
    top30 = abc_df.head(30)
    colors = top30['class'].map({'A': '#28a745', 'B': '#ffc107', 'C': '#dc3545'})
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=top30['item_name'].str[:20], y=top30['revenue']/1e6,
                        name='Doanh thu (M₽)', marker_color=colors.tolist()), secondary_y=False)
    fig.add_trace(go.Scatter(x=top30['item_name'].str[:20], y=top30['cum_pct'],
                            name='% Tích lũy', line=dict(color='#2E86AB', width=3)), secondary_y=True)
    fig.add_hline(y=80, line_dash="dash", line_color="green", secondary_y=True)
    fig.add_hline(y=95, line_dash="dash", line_color="orange", secondary_y=True)
    fig.update_layout(title='Biểu đồ Pareto (Top 30)', xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

# === TAB 6: MODEL INFO ===
with tabs[5]:
    st.header("🧠 Thông tin ML Model")
    
    if available_models:
        st.success(f"✅ Đã load **{len(available_models)}** models từ `{model_dir}`")
        
        for name, model in available_models.items():
            with st.expander(f"📦 {name} Model", expanded=(name == selected_model_type)):
                st.write(f"**File:** `{model_dir}/{name.lower()}_model.h5`")
                
                # Model architecture
                st.write("**Kiến trúc:**")
                summary_list = []
                model.summary(print_fn=lambda x: summary_list.append(x))
                st.code('\n'.join(summary_list), language='text')
                
                # Performance if config exists
                if model_config and 'results' in model_config and name in model_config['results']:
                    r = model_config['results'][name]
                    col1, col2 = st.columns(2)
                    col1.metric("Train RMSE", f"{r['train_rmse']:.4f}")
                    col2.metric("Val RMSE", f"{r['val_rmse']:.4f}")
    else:
        st.warning("⚠️ Không tìm thấy model nào")
        st.markdown(f"""
        ### 📋 Hướng dẫn export model từ notebook
        
        **Bước 1:** Mở `demand_forecasting_tf2.ipynb` và chạy hết các cells
        
        **Bước 2:** Thêm cell sau vào cuối notebook (sau khi train xong):
        
        ```python
        import os
        os.makedirs('./models', exist_ok=True)
        
        # Lưu các models
        model_mlp.save('./models/mlp_model.h5')
        model_cnn.save('./models/cnn_model.h5')
        model_lstm.save('./models/lstm_model.h5')
        model_cnn_lstm.save('./models/cnn_lstm_model.h5')
        
        print("✅ Đã lưu models!")
        ```
        
        **Bước 3:** Chạy cell và restart app
        
        ---
        
        📁 **Thư mục models hiện tại:** `{model_dir}`
        
        💡 **Tip:** App vẫn hoạt động với **Rule-based** khi không có model
        """)

# === FOOTER ===
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
🎓 <b>Đồ án môn Deep Learning - UIT</b><br>
📊 Predict Future Sales (Kaggle) | 🤖 ML Models: LSTM, CNN, MLP, CNN-LSTM<br>
Made with ❤️ using Streamlit & TensorFlow
</div>
""", unsafe_allow_html=True)
