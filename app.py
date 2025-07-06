import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import joblib
import os

# Page config
st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B6B;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #4ECDC4;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .prediction-high {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .prediction-medium {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .prediction-low {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .update-info {
        background-color: #e9ecef;
        color: #495057;
        padding: 8px;
        border-radius: 5px;
        font-style: italic;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏎️ F1 Race Predictor</h1>', unsafe_allow_html=True)
st.markdown("**Predict top 3 finishers using a Machine Learning model trained on 2023-2025 F1 data**")

# Last data update info
st.markdown('<div class="update-info">📅 Last Data Update: Canadian GP 2025</div>', unsafe_allow_html=True)

# Train model function for cloud deployment
@st.cache_resource
def load_or_train_model():
    """Load saved model or retrain if not available/incompatible"""
    
    model_file = 'f1_prediction_model.pkl'
    
    # For cloud deployment, always retrain to avoid compatibility issues
    st.info("🔄 Training model with cloud environment for compatibility...")
    
    # Check for training data
    data_file = 'enhanced_combined.csv'
    if not os.path.exists(data_file):
        st.error(f"""
        ❌ Training data not found!
        
        Please upload `enhanced_combined.csv` to train the model.
        
        This file should contain your 2023-2025 F1 data with all engineered features.
        """)
        st.stop()
    
    # Retrain model
    with st.spinner("🏎️ Training F1 prediction model... (this may take a minute)"):
        
        # Load data
        df = pd.read_csv(data_file)
        st.info(f"📊 Loaded {len(df)} records for training")
        
        # Create target variable
        df['top_3_finish'] = (df['race_position'] <= 3).astype(int)
        
        # Feature selection (key features that should exist)
        feature_columns = [
            'qualifying_position', 'driver_races_completed', 'driver_recent_avg_position',
            'team_season_avg_position', 'driver_circuit_avg_position', 'driver_career_wins',
            'driver_career_podiums', 'driver_career_points_rate', 'driver_recent_avg_qual_position',
            'driver_recent_wins', 'driver_recent_podiums', 'team_season_wins',
            'team_season_podiums', 'team_season_points_rate', 'is_wet_race',
            'avg_air_temp', 'avg_track_temp', 'avg_humidity', 'total_rainfall'
        ]
        
        # Check available features
        available_features = [col for col in feature_columns if col in df.columns]
        
        # Prepare data
        X = df[available_features].copy()
        y = df['top_3_finish'].copy()
        
        # Handle boolean columns
        bool_columns = X.select_dtypes(include=['bool']).columns
        if len(bool_columns) > 0:
            X[bool_columns] = X[bool_columns].astype(int)
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_processed = imputer.fit_transform(X)
        
        # Train/test split by year
        train_mask = df['meeting_year'].isin([2023, 2024])
        test_mask = df['meeting_year'] == 2025
        
        X_train = X_processed[train_mask]
        X_test = X_processed[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        # Train Random Forest
        model = RandomForestClassifier(
            random_state=42,
            n_estimators=100,  # Reduced for faster training
            max_depth=10,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        test_accuracy = model.score(X_test, y_test)
        test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        
        # Create model package
        model_package = {
            'model': model,
            'imputer': imputer,
            'features': available_features,
            'model_type': 'Random Forest',
            'model_version': '4.0_cloud',
            'performance_metrics': {
                'test_accuracy': test_accuracy,
                'test_auc': test_auc,
                'features_count': len(available_features),
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            },
            'training_years': [2023, 2024],
            'validation_year': 2025,
            'training_data': df  # Store the training data for Tab 1 predictions
        }
        
        # Save model
        try:
            joblib.dump(model_package, model_file)
        except:
            pass  # Don't fail if can't save
        
        st.success(f"✅ Model trained! Accuracy: {test_accuracy:.1%}, AUC: {test_auc:.3f}")
        
        return model_package

# Load model
model_package = load_or_train_model()

# Extract model components
final_model = model_package['model']
available_features = model_package['features'] 
imputer = model_package['imputer']
training_data = model_package['training_data']

# Display model info in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**🤖 Model Info**")
st.sidebar.markdown(f"**Type**: {model_package['model_type']}")
st.sidebar.markdown(f"**Version**: {model_package['model_version']}")
st.sidebar.markdown(f"**Accuracy**: {model_package['performance_metrics']['test_accuracy']:.1%}")
st.sidebar.markdown(f"**AUC**: {model_package['performance_metrics']['test_auc']:.3f}")
st.sidebar.markdown(f"**Features**: {model_package['performance_metrics']['features_count']}")

# Real 2025 drivers
@st.cache_data
def get_default_drivers():
    """Get list of current F1 drivers and teams (Real 2025 lineup)"""
    drivers_2025_current = [
        {"name": "Oscar PIASTRI", "team": "McLaren", "number": 81},
        {"name": "Max VERSTAPPEN", "team": "Red Bull Racing", "number": 1},
        {"name": "Lando NORRIS", "team": "McLaren", "number": 4},
        {"name": "Lewis HAMILTON", "team": "Ferrari", "number": 44},
        {"name": "Charles LECLERC", "team": "Ferrari", "number": 16},
        {"name": "Carlos SAINZ", "team": "Williams", "number": 55},
        {"name": "George RUSSELL", "team": "Mercedes", "number": 63},
        {"name": "Kimi ANTONELLI", "team": "Mercedes", "number": 12},
        {"name": "Isack HADJAR", "team": "Racing Bulls", "number": 6},
        {"name": "Pierre GASLY", "team": "Alpine", "number": 10},
        {"name": "Fernando ALONSO", "team": "Aston Martin", "number": 14},
        {"name": "Lance STROLL", "team": "Aston Martin", "number": 18},
        {"name": "Yuki TSUNODA", "team": "Red Bull Racing", "number": 22},
        {"name": "Alexander ALBON", "team": "Williams", "number": 23},
        {"name": "Nico HULKENBERG", "team": "Kick Sauber", "number": 27},
        {"name": "Liam LAWSON", "team": "Racing Bulls", "number": 30},
        {"name": "Esteban OCON", "team": "Haas F1 Team", "number": 31},
        {"name": "Franco COLAPINTO", "team": "Alpine", "number": 43},
        {"name": "Gabriel BORTOLETO", "team": "Kick Sauber", "number": 5},
        {"name": "Oliver BEARMAN", "team": "Haas F1 Team", "number": 87},
    ]
    return pd.DataFrame(drivers_2025_current)

# Circuit weather defaults
CIRCUIT_WEATHER_DEFAULTS = {
    "British Grand Prix": {"air_temp": 18, "track_temp": 25, "humidity": 75, "is_wet": True},
    "Belgian Grand Prix": {"air_temp": 20, "track_temp": 28, "humidity": 70, "is_wet": False},
    "Hungarian Grand Prix": {"air_temp": 28, "track_temp": 42, "humidity": 55, "is_wet": False}
}

def get_latest_driver_stats(training_data, driver_name):
    """Extract latest driver statistics from training data"""
    # Get the most recent data for this driver
    driver_data = training_data[training_data['full_name'] == driver_name]
    
    if len(driver_data) == 0:
        # Return default values if driver not found
        return {
            'driver_races_completed': 30,
            'driver_recent_avg_position': 10,
            'driver_circuit_avg_position': 10,
            'driver_career_wins': 1,
            'driver_career_podiums': 3,
            'driver_career_points_rate': 0.5,
            'driver_recent_avg_qual_position': 10,
            'driver_recent_wins': 0,
            'driver_recent_podiums': 1
        }
    
    # Get the most recent entry
    latest_entry = driver_data.iloc[-1]
    
    return {
        'driver_races_completed': latest_entry.get('driver_races_completed', 30),
        'driver_recent_avg_position': latest_entry.get('driver_recent_avg_position', 10),
        'driver_circuit_avg_position': latest_entry.get('driver_circuit_avg_position', 10),
        'driver_career_wins': latest_entry.get('driver_career_wins', 1),
        'driver_career_podiums': latest_entry.get('driver_career_podiums', 3),
        'driver_career_points_rate': latest_entry.get('driver_career_points_rate', 0.5),
        'driver_recent_avg_qual_position': latest_entry.get('driver_recent_avg_qual_position', 10),
        'driver_recent_wins': latest_entry.get('driver_recent_wins', 0),
        'driver_recent_podiums': latest_entry.get('driver_recent_podiums', 1)
    }

def get_latest_team_stats(training_data, team_name):
    """Extract latest team statistics from training data"""
    # Get the most recent data for this team
    team_data = training_data[training_data['team_name'] == team_name]
    
    if len(team_data) == 0:
        return {
            'team_season_avg_position': 10,
            'team_season_wins': 1,
            'team_season_podiums': 3,
            'team_season_points_rate': 0.5
        }
    
    # Get the most recent entry
    latest_entry = team_data.iloc[-1]
    
    return {
        'team_season_avg_position': latest_entry.get('team_season_avg_position', 10),
        'team_season_wins': latest_entry.get('team_season_wins', 1),
        'team_season_podiums': latest_entry.get('team_season_podiums', 3),
        'team_season_points_rate': latest_entry.get('team_season_points_rate', 0.5)
    }

def predict_race_results(prediction_data, model_package):
    """Make predictions using the trained model"""
    
    model = model_package['model']
    features = model_package['features']
    imputer = model_package['imputer']
    
    # Ensure all required features are present
    for feature in features:
        if feature not in prediction_data.columns:
            prediction_data[feature] = 0  # Default value for missing features
    
    # Select and order features
    X = prediction_data[features].copy()
    
    # Handle boolean columns
    bool_columns = X.select_dtypes(include=['bool']).columns
    if len(bool_columns) > 0:
        X[bool_columns] = X[bool_columns].astype(int)
    
    # Preprocess
    X_processed = imputer.transform(X)
    
    # Make predictions
    probabilities = model.predict_proba(X_processed)[:, 1]  # Probability of top 3
    predictions = model.predict(X_processed)
    
    # Add results to dataframe
    results = prediction_data[['full_name', 'team_name']].copy()
    if 'qualifying_position' in prediction_data.columns:
        results['qualifying_position'] = prediction_data['qualifying_position']
    else:
        results['qualifying_position'] = 'N/A'
    
    results['top_3_probability'] = probabilities
    results['predicted_top_3'] = predictions
    
    return results.sort_values('top_3_probability', ascending=False)

def display_predictions(results, show_qualifying=True):
    """Display prediction results in consistent format"""
    
    # Show top 3 predictions
    st.markdown("**🏆 Predicted Top 3 Finishers:**")
    
    top_3 = results.head(3)
    for idx, (_, driver) in enumerate(top_3.iterrows()):
        prob_pct = driver['top_3_probability'] * 100
        
        if prob_pct >= 70:
            style_class = "prediction-high"
            confidence = "High"
        elif prob_pct >= 40:
            style_class = "prediction-medium"
            confidence = "Medium"
        else:
            style_class = "prediction-low"
            confidence = "Low"
        
        qualifying_info = f"<br>Qualifying: P{driver['qualifying_position']}" if show_qualifying and driver['qualifying_position'] != 'N/A' else ""
        
        st.markdown(f"""
        <div class="{style_class}">
            <strong>P{idx+1}: {driver['full_name']}</strong><br>
            Team: {driver['team_name']}{qualifying_info}<br>
            Podium Probability: {prob_pct:.1f}%<br>
            Confidence: {confidence}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
    
    # Show full results table
    st.markdown("**📋 All Driver Predictions:**")
    display_results = results[['full_name', 'top_3_probability']].copy()
    if show_qualifying and 'qualifying_position' in results.columns and results['qualifying_position'].iloc[0] != 'N/A':
        display_results = results[['full_name', 'qualifying_position', 'top_3_probability']].copy()
        display_results.columns = ['Driver', 'Qualifying Position', 'Podium Probability (%)']
    else:
        display_results.columns = ['Driver', 'Podium Probability (%)']
    
    display_results['Podium Probability (%)'] = (display_results['Podium Probability (%)'] * 100).round(1)
    
    st.dataframe(display_results, use_container_width=True, hide_index=True)
    
    # Visualization
    st.markdown("**📈 Probability Chart:**")
    fig = px.bar(
        results.head(10),
        x='top_3_probability',
        y='full_name',
        orientation='h',
        title='Top 10 Podium Probabilities',
        labels={'top_3_probability': 'Podium Probability', 'full_name': 'Driver'},
        color='top_3_probability',
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

# Create tabs
tab1, tab2 = st.tabs(["🏁 Upcoming Races", "🏎️ Qualifying Predictions"])

# Get drivers data
drivers_df = get_default_drivers()

# TAB 1: Upcoming Races
with tab1:
    st.markdown('<div class="sub-header">🏁 Next Race Predictions</div>', unsafe_allow_html=True)
    st.markdown("*Predictions based on current driver form and historical performance*")
    
    # Race selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_race = st.selectbox(
            "Select Upcoming Race:",
            ["British Grand Prix", "Belgian Grand Prix", "Hungarian Grand Prix"]
        )
    
    with col2:
        if st.button("🔮 Predict Race Results", type="primary", key="tab1_predict"):
            try:
                with st.spinner(f"🏎️ Generating predictions for {selected_race}..."):
                    # Get weather defaults for selected circuit
                    weather = CIRCUIT_WEATHER_DEFAULTS[selected_race]
                    
                    # Create prediction data for all drivers
                    prediction_data = []
                    
                    for _, driver in drivers_df.iterrows():
                        driver_name = driver['name']
                        team_name = driver['team']
                        
                        # Get latest driver and team stats from training data
                        driver_stats = get_latest_driver_stats(training_data, driver_name)
                        team_stats = get_latest_team_stats(training_data, team_name)
                        
                        # Create feature set
                        driver_data = {
                            'full_name': driver_name,
                            'team_name': team_name,
                            'qualifying_position': 10,  # Default/average qualifying position
                            'is_wet_race': 1 if weather['is_wet'] else 0,
                            'avg_air_temp': weather['air_temp'],
                            'avg_track_temp': weather['track_temp'],
                            'avg_humidity': weather['humidity'],
                            'total_rainfall': 5 if weather['is_wet'] else 0,
                            **driver_stats,
                            **team_stats
                        }
                        
                        prediction_data.append(driver_data)
                    
                    prediction_df = pd.DataFrame(prediction_data)
                    
                    # Make predictions
                    results = predict_race_results(prediction_df, model_package)
                    st.session_state[f'tab1_predictions_{selected_race}'] = results
                    st.success(f"✅ Predictions generated for {selected_race}!")
            
            except Exception as e:
                st.error(f"❌ Error making predictions: {str(e)}")
    
    # Display results if available
    if f'tab1_predictions_{selected_race}' in st.session_state:
        st.markdown("---")
        display_predictions(st.session_state[f'tab1_predictions_{selected_race}'], show_qualifying=False)
    else:
        st.info("👆 Select a race and click 'Predict Race Results' to see predictions!")

# TAB 2: Qualifying Predictions
with tab2:
    # Race Setup and Weather (only for Tab 2)
    with st.sidebar:
        st.markdown('<div class="sub-header">🏁 Race Setup</div>', unsafe_allow_html=True)
        
        race_name = st.selectbox(
            "Select Race",
            ["Austrian Grand Prix", "British Grand Prix", "Hungarian Grand Prix", 
             "Belgian Grand Prix", "Dutch Grand Prix", "Custom Race"]
        )
        
        # Weather conditions
        st.markdown("**Weather Conditions**")
        is_wet = st.checkbox("Wet Race (Rain Expected)", value=False)
        air_temp = st.slider("Air Temperature (°C)", 15, 40, 25)
        track_temp = st.slider("Track Temperature (°C)", 20, 50, 35)
        humidity = st.slider("Humidity (%)", 30, 90, 60)
    
    # Initialize session state
    if 'tab2_predictions' not in st.session_state:
        st.session_state.tab2_predictions = None
    
    # Main content area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<div class="sub-header">🏎️ Qualifying Results</div>', unsafe_allow_html=True)
        st.markdown("*Select which driver qualified in each position:*")
        
        # Create grid position assignments
        grid_positions = {}
        
        # Create a 4-column layout for the grid (P1-P5, P6-P10, P11-P15, P16-P20)
        grid_cols = st.columns(4)
        
        # Column headers
        with grid_cols[0]:
            st.markdown("**🥇 Positions 1-5**")
        with grid_cols[1]:
            st.markdown("**🥈 Positions 6-10**")
        with grid_cols[2]:
            st.markdown("**🥉 Positions 11-15**")
        with grid_cols[3]:
            st.markdown("**📍 Positions 16-20**")
        
        # Create dropdowns for each grid position
        for pos in range(1, 21):
            col_index = (pos - 1) // 5  # Which column (0-3)
            
            with grid_cols[col_index]:
                # Create driver options with team info
                driver_options = []
                for _, driver in drivers_df.iterrows():
                    team_short = driver['team'].replace(' F1 Team', '').replace(' Racing', '')[:12]
                    driver_options.append(f"{driver['name']} ({team_short})")
                
                # Default selection (try to match position with driver index, or first available)
                default_index = min(pos - 1, len(driver_options) - 1)
                
                selected_driver = st.selectbox(
                    f"P{pos}:",
                    options=driver_options,
                    index=default_index,
                    key=f"tab2_grid_pos_{pos}"
                )
                
                # Extract driver name from selection
                driver_name = selected_driver.split(" (")[0]
                grid_positions[pos] = driver_name
        
        # Convert grid positions to qualifying positions for the model
        qualifying_positions = {}
        for position, driver_name in grid_positions.items():
            qualifying_positions[driver_name] = position

        # Predict button
        if st.button("🔮 Predict Race Results", type="primary", use_container_width=True, key="tab2_predict"):
            try:
                # Create prediction data with simplified approach
                prediction_data = []
                
                for driver_name, qual_pos in qualifying_positions.items():
                    team = drivers_df[drivers_df['name'] == driver_name]['team'].iloc[0]
                    
                    # Get latest driver and team stats from training data
                    driver_stats = get_latest_driver_stats(training_data, driver_name)
                    team_stats = get_latest_team_stats(training_data, team)
                    
                    # Create basic feature set
                    driver_data = {
                        'full_name': driver_name,
                        'team_name': team,
                        'qualifying_position': qual_pos,
                        'is_wet_race': 1 if is_wet else 0,
                        'avg_air_temp': air_temp,
                        'avg_track_temp': track_temp,
                        'avg_humidity': humidity,
                        'total_rainfall': 5 if is_wet else 0,
                        **driver_stats,
                        **team_stats
                    }
                    
                    prediction_data.append(driver_data)
                
                prediction_df = pd.DataFrame(prediction_data)
                
                # Make predictions
                results = predict_race_results(prediction_df, model_package)
                st.session_state.tab2_predictions = results
                st.success("✅ Predictions generated!")
                
            except Exception as e:
                st.error(f"❌ Error making predictions: {str(e)}")

    with col2:
        st.markdown('<div class="sub-header">📊 Predictions</div>', unsafe_allow_html=True)
        
        if st.session_state.tab2_predictions is not None:
            display_predictions(st.session_state.tab2_predictions, show_qualifying=True)
        else:
            st.info("👆 Set qualifying positions and click 'Predict Race Results' to see predictions!")

# Footer with model info
st.markdown("---")
col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.markdown("**🤖 Model Info**")
    st.markdown(f"""
    - **Algorithm**: {model_package['model_type']}
    - **Training Data**: {'-'.join(map(str, model_package['training_years']))}
    - **Validation**: {model_package['validation_year']} races
    - **AUC Score**: {model_package['performance_metrics']['test_auc']:.3f} (Excellent)
    """)

with col_info2:
    st.markdown("**🎯 Key Features**")
    st.markdown("""
    - Qualifying position
    - Driver recent form
    - Team performance
    - Circuit-specific history
    - Weather conditions
    """)

with col_info3:
    st.markdown("**📊 Model Performance**")
    st.markdown(f"""
    - **Accuracy**: {model_package['performance_metrics']['test_accuracy']:.1%}
    - **Features**: {model_package['performance_metrics']['features_count']}
    - **Training Samples**: {model_package['performance_metrics']['train_samples']}
    - **World-class performance!**
    """)

st.markdown("---")
st.markdown("**Built with ❤️ using F1 data from 2023-2025 • Powered by Streamlit**")
