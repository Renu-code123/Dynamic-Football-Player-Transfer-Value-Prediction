import streamlit as st
import requests
import json

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="TransferIQ - Player Value Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS FOR MODERN STYLING
# =====================================================
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .header-title {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .header-subtitle {
        color: #e0e7ff;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 0;
    }
    
    /* Input section styling */
    .input-section {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    
    /* Result card styling */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 2rem 0;
    }
    
    .result-label {
        color: #e0e7ff;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .result-value {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        color: #6b7280;
        font-size: 0.9rem;
        border-top: 1px solid #e5e7eb;
    }
    
    /* Section headers */
    .section-header {
        color: #1f2937;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    
    /* Info box */
    .info-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER SECTION
# =====================================================
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">⚽ TransferIQ</h1>
        <p class="header-subtitle">AI-Powered Football Player Transfer Value Prediction</p>
    </div>
""", unsafe_allow_html=True)

# =====================================================
# INTRODUCTION SECTION
# =====================================================
st.markdown("""
    <div class="info-box">
        <strong>About TransferIQ:</strong> This advanced machine learning system predicts football player 
        transfer market values based on comprehensive performance metrics, physical attributes, and career statistics. 
        Enter player data below to get an instant valuation.
    </div>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR - API CONFIGURATION
# =====================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input(
        "FastAPI Backend URL",
        value="http://localhost:8000/predict",
        help="Enter the URL of your FastAPI prediction endpoint"
    )
    
    st.markdown("---")
    st.markdown("### 📊 About the Model")
    st.info("""
    The model uses 16 key features including:
    - Performance metrics (goals, assists)
    - Game statistics (matches, minutes)
    - Discipline records (cards, fouls)
    - Physical attributes (height)
    - Career data (injuries, position)
    """)

# =====================================================
# MAIN INPUT FORM
# =====================================================
st.markdown('<p class="section-header">📋 Player Information</p>', unsafe_allow_html=True)

# Create tabs for organized input
tab1, tab2, tab3, tab4 = st.tabs(["⚽ Performance", "📊 Game Stats", "⚠️ Discipline", "👤 Physical & Career"])

with tab1:
    st.markdown("### Performance Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        total_goals = st.number_input(
            "Total Goals",
            min_value=0.0,
            value=39.0,
            step=1.0,
            help="Total career goals scored"
        )
        total_assists = st.number_input(
            "Total Assists",
            min_value=0.0,
            value=23.0,
            step=1.0,
            help="Total career assists"
        )
        total_penalty_goals = st.number_input(
            "Penalty Goals",
            min_value=0.0,
            value=3.0,
            step=1.0,
            help="Goals scored from penalties"
        )
    
    with col2:
        total_own_goals = st.number_input(
            "Own Goals",
            min_value=0.0,
            value=1.0,
            step=1.0,
            help="Total own goals"
        )

with tab2:
    st.markdown("### Game Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        total_matches = st.number_input(
            "Total Matches",
            min_value=0.0,
            value=24.0,
            step=1.0,
            help="Total matches played"
        )
        total_minutes_played = st.number_input(
            "Minutes Played",
            min_value=0.0,
            value=7865.0,
            step=100.0,
            help="Total minutes on the pitch"
        )
        total_nb_on_pitch = st.number_input(
            "Times on Pitch",
            min_value=0.0,
            value=268.0,
            step=1.0,
            help="Number of times started on pitch"
        )
        total_nb_in_group = st.number_input(
            "Times in Squad",
            min_value=0.0,
            value=320.0,
            step=1.0,
            help="Number of times in matchday squad"
        )
    
    with col2:
        total_subed_in = st.number_input(
            "Substituted In",
            min_value=0.0,
            value=49.0,
            step=1.0,
            help="Times substituted into match"
        )
        total_subed_out = st.number_input(
            "Substituted Out",
            min_value=0.0,
            value=59.0,
            step=1.0,
            help="Times substituted out of match"
        )

with tab3:
    st.markdown("### Discipline Records")
    col1, col2 = st.columns(2)
    
    with col1:
        total_yellow_cards = st.number_input(
            "Yellow Cards",
            min_value=0.0,
            value=34.0,
            step=1.0,
            help="Total yellow cards received"
        )
        total_second_yellow_cards = st.number_input(
            "Second Yellow Cards",
            min_value=0.0,
            value=1.0,
            step=1.0,
            help="Second yellow cards (red card offenses)"
        )
    
    with col2:
        total_direct_red_cards = st.number_input(
            "Direct Red Cards",
            min_value=0.0,
            value=1.0,
            step=1.0,
            help="Direct red cards received"
        )

with tab4:
    st.markdown("### Physical Attributes & Career Data")
    col1, col2 = st.columns(2)
    
    with col1:
        height = st.number_input(
            "Height (cm)",
            min_value=0.0,
            value=165.0,
            step=1.0,
            help="Player height in centimeters"
        )
        career_total_injuries = st.number_input(
            "Career Injuries",
            min_value=0.0,
            value=5.0,
            step=1.0,
            help="Total career injuries"
        )
    
    with col2:
        main_position_encoded_midfield = st.selectbox(
            "Position: Midfield?",
            options=[0, 1],
            index=0,
            help="Is the player's main position midfielder? (1=Yes, 0=No)"
        )

# =====================================================
# PREDICTION BUTTON & RESULT
# =====================================================
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🚀 Predict Transfer Value", use_container_width=True, type="primary")

if predict_button:
    # Prepare payload
    payload = {
        "total_assists": total_assists,
        "total_nb_in_group": total_nb_in_group,
        "total_matches": total_matches,
        "total_subed_in": total_subed_in,
        "total_penalty_goals": total_penalty_goals,
        "total_goals": total_goals,
        "career_total_injuries": career_total_injuries,
        "total_yellow_cards": total_yellow_cards,
        "total_nb_on_pitch": total_nb_on_pitch,
        "total_minutes_played": total_minutes_played,
        "height": height,
        "total_subed_out": total_subed_out,
        "total_own_goals": total_own_goals,
        "total_second_yellow_cards": total_second_yellow_cards,
        "total_direct_red_cards": total_direct_red_cards,
        "main_position_encoded_midfield": main_position_encoded_midfield
    }
    
    # Make API request
    try:
        with st.spinner("🔮 Calculating player value..."):
            response = requests.post(api_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                predicted_value = result["predicted_market_value"]
                
                # Display result in a beautiful card
                st.markdown(f"""
                    <div class="result-card">
                        <p class="result-label">Predicted Transfer Market Value</p>
                        <h1 class="result-value">€{predicted_value:,.2f}</h1>
                    </div>
                """, unsafe_allow_html=True)
                
                # Additional insights
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Value in Millions",
                        value=f"€{predicted_value/1_000_000:.2f}M"
                    )
                
                with col2:
                    st.metric(
                        label="Goals per Match",
                        value=f"{total_goals/total_matches if total_matches > 0 else 0:.2f}"
                    )
                
                with col3:
                    st.metric(
                        label="Minutes per Match",
                        value=f"{total_minutes_played/total_matches if total_matches > 0 else 0:.0f}"
                    )
                
                st.success("✅ Prediction completed successfully!")
                
            else:
                st.error(f"❌ Error: {response.status_code} - {response.text}")
                
    except requests.exceptions.ConnectionError:
        st.error("❌ Connection Error: Unable to reach the FastAPI backend. Please ensure it's running.")
    except requests.exceptions.Timeout:
        st.error("❌ Timeout Error: The request took too long. Please try again.")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown("""
    <div class="footer">
        <p><strong>This project is made under Infosys Springboard Virtual Internship 6.0</strong></p>
        <p style="margin-top: 0.5rem; color: #9ca3af;">TransferIQ</p>
    </div>
""", unsafe_allow_html=True)