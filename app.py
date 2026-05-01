import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ✅ Only keep this import
from langchain_community.chat_models import ChatOllama

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="NeuroFinance Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Custom CSS for Professional UI
# -----------------------------
st.markdown("""
<style>
    /* Import professional font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Main background - professional neutral */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Header styling */
    .header-container {
        text-align: center;
        padding: 2.5rem 0;
        margin-bottom: 2rem;
    }

    .header-title {
        font-size: 3.2rem;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        letter-spacing: -0.5px;
    }

    .header-subtitle {
        font-size: 1.15rem;
        color: rgba(255, 255, 255, 0.9);
        margin-top: 0.5rem;
        font-weight: 400;
    }

    /* Card styling - modern white cards */
    .prediction-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
    }

    .prediction-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
    }

    .metric-label {
        font-size: 0.75rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }

    /* Chat container */
    .chat-container {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }

    .chat-message {
        background: #f3f4f6;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0.8rem 0;
        border-left: 4px solid #667eea;
        color: #1f2937;
        line-height: 1.5;
    }

    .chat-message-user {
        background: #dbeafe;
        border-left: 4px solid #3b82f6;
        color: #1f2937;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ffffff;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 10px rgba(16, 185, 129, 0.3);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
        transform: translateY(-1px);
    }

    /* Section headers */
    .section-header {
        color: #ffffff;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
    }

    /* Info box */
    .info-box {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.25rem;
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .info-box p {
        color: rgba(255, 255, 255, 0.95);
        line-height: 1.6;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load and Train Models (Cached)
# -----------------------------
@st.cache_resource
def load_models():
    df = pd.read_csv('cleaned1012_dataset.csv')

    target_columns = [
        'Inflation_YoY',
        'Oil_Price_USD_Barrel',
        'Exchange_Rate_PKR_USD',
        'Interest_Rate',
        'Money_Supply_M2_Billion',
        'pkr_to_oneDollar',
        'Gold_Price_In_Dolars',
        'Petrol_Price',
        'Minimum_Wage_PKR'
    ]

    models = {}
    existing_targets = [col for col in target_columns if col in df.columns]

    for col in existing_targets:
        valid_data = df.dropna(subset=[col, 'Year'])
        X_train = valid_data[['Year']].values
        y_train = valid_data[col].values

        model = LinearRegression()
        model.fit(X_train, y_train)
        models[col] = model

    return models

# -----------------------------
# Prediction Function
# -----------------------------
def get_predictions(target_year, models):
    year_input = np.array([[target_year]])
    predictions = {}

    for name, model in models.items():
        pred = model.predict(year_input)[0]
        predictions[name] = round(pred, 2)

    return predictions

# -----------------------------
# LLM Setup
# -----------------------------
@st.cache_resource
def get_llm():
    return ChatOllama(model="llama3:latest")

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🧠 NeuroFinance Predictor</h1>
    <p class="header-subtitle">AI-Powered Economic Forecasting for Pakistan</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Main Content
# -----------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📊 Prediction Controls")

    target_year = st.number_input(
        "Enter Year for Prediction",
        min_value=2025,
        max_value=2050,
        value=2028,
        step=1,
        help="Select the year you want to forecast"
    )

    analyze_button = st.button("🚀 Generate Forecast", use_container_width=True)

with col2:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### 📈 Quick Stats")
    st.markdown("""
    - **9 Economic Indicators** tracked
    - **Machine Learning** powered predictions
    - **AI Analysis** from Llama3
    - Real-time forecasting
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Results Display
# -----------------------------
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
    st.session_state.explanation = None
    st.session_state.year = None

if analyze_button:
    with st.spinner("🔮 Analyzing economic trends..."):
        models = load_models()
        predictions = get_predictions(target_year, models)
        st.session_state.predictions = predictions
        st.session_state.year = target_year

        # Generate AI explanation
        llm = get_llm()

        formatted_data = "\n".join([f"{k}: {v}" for k, v in predictions.items()])

        final_prompt = f"""
You are an expert economist analyzing Pakistan's economy.

PREDICTED INDICATORS FOR {target_year}:
{formatted_data}

Provide a concise analysis covering:
1. Key trends and what they mean
2. Impact on ordinary citizens
3. Relationship between inflation, wages, and currency
4. One actionable insight

Keep it under 200 words.
"""

        with st.spinner("🤖 AI generating insights..."):
            response = llm.invoke(final_prompt)
            st.session_state.explanation = response.content

# -----------------------------
# Display Predictions
# -----------------------------
if st.session_state.predictions:
    st.markdown("---")
    st.markdown(f'<p class="section-header">📊 Economic Forecast for {st.session_state.year}</p>', unsafe_allow_html=True)

    # Create metric cards
    cols = st.columns(3)

    # Color-coded metrics based on type
    metrics_display = {
        "💵 PKR/USD Rate": (f"{st.session_state.predictions.get('pkr_to_oneDollar', 'N/A')}", "#2980b9"),
        "🛢️ Oil Price": (f"${st.session_state.predictions.get('Oil_Price_USD_Barrel', 'N/A')}", "#e67e22"),
        "📈 Inflation": (f"{st.session_state.predictions.get('Inflation_YoY', 'N/A')}%", "#c0392b"),
        "💰 Interest Rate": (f"{st.session_state.predictions.get('Interest_Rate', 'N/A')}%", "#8e44ad"),
        "🏦 Money Supply": (f"{st.session_state.predictions.get('Money_Supply_M2_Billion', 'N/A')} B", "#16a085"),
        "📉 Exchange Rate": (f"{st.session_state.predictions.get('Exchange_Rate_PKR_USD', 'N/A')}", "#2980b9"),
        "🥇 Gold Price": (f"${st.session_state.predictions.get('Gold_Price_In_Dolars', 'N/A')}", "#f39c12"),
        "⛽ Petrol Price": (f"{st.session_state.predictions.get('Petrol_Price', 'N/A')}", "#d35400"),
        "💼 Min Wage": (f"Rs. {st.session_state.predictions.get('Minimum_Wage_PKR', 'N/A'):,.0f}", "#27ae60")
    }

    for idx, (label, (value, color)) in enumerate(metrics_display.items()):
        with cols[idx % 3]:
            st.markdown(f"""
            <div class="prediction-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color: {color};">{value}</div>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------
# AI Chat Section
# -----------------------------
st.markdown("---")
st.markdown('<p class="section-header">🤖 AI Economic Advisor</p>', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message chat-message-user">
            <strong>👤 You:</strong><br>{message["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message">
            <strong>🧠 NeuroFinance AI:</strong><br>{message["content"]}
        </div>
        """, unsafe_allow_html=True)

# Chat input
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
col_chat1, col_chat2 = st.columns([5, 1])

with col_chat1:
    user_input = st.text_input(
        "Ask about the economy",
        placeholder="e.g., How will inflation affect purchasing power?",
        key="chat_input",
        label_visibility="collapsed"
    )

with col_chat2:
    send_button = st.button("Send", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Process chat input
if send_button and user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get AI response
    llm = get_llm()

    # Build context-aware prompt
    context = ""
    if st.session_state.predictions:
        context = f"Current predictions for {st.session_state.year}: {st.session_state.predictions}"

    chat_prompt = f"""
You are NeuroFinance AI, an expert economic advisor for Pakistan.

{context}

User question: {user_input}

Provide a helpful, accurate response in 2-3 sentences.
"""

    with st.spinner("AI thinking..."):
        response = llm.invoke(chat_prompt)
        st.session_state.messages.append({"role": "assistant", "content": response.content})

    st.rerun()
