"""
Streamlit app for NYC Taxi Fare Prediction
"""

import streamlit as st
import requests
from datetime import datetime, time
import json


# ============================================
# CONFIGURATION
# ============================================

API_URL = "http://localhost:8000"  # FastAPI endpoint

# Page config
st.set_page_config(
    page_title="NYC Taxi Fare Predictor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================
# HELPER FUNCTIONS
# ============================================


def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def make_prediction(trip_data):
    """
    Call API to make prediction

    Args:
        trip_data: Dictionary with trip information

    Returns:
        Prediction response or None if error
    """
    try:
        response = requests.post(
            f"{API_URL}/predict", json=trip_data, timeout=5
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Cannot connect to API. Make sure it's running on port 8000"
        )
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None


# ============================================
# MAIN APP
# ============================================


def main():
    """Main Streamlit application"""

    # Header
    st.title("🚕 NYC Taxi Fare Predictor")
    st.markdown(
        """
        Welcome to the NYC Taxi Fare Prediction app! 
        Enter trip details below to get an estimated fare.
        """
    )

    # Check API status
    api_status = check_api_health()

    # Status indicator
    col1, col2 = st.columns([3, 1])
    with col2:
        if api_status:
            st.success("🟢 API Online")
        else:
            st.error("🔴 API Offline")
            st.warning(
                "Please start the API:\n```bash\n"
                "uvicorn src.api.app:app --reload\n```"
            )
            st.stop()

    st.markdown("---")

    # ============================================
    # INPUT FORM
    # ============================================

    st.subheader("📝 Trip Details")

    # Create two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        # Vendor ID
        vendor_id = st.selectbox(
            "Vendor",
            options=[1, 2],
            format_func=lambda x: "Creative Mobile Technologies"
            if x == 1
            else "VeriFone Inc.",
            help="The taxi company",
        )

        # Passenger count
        passenger_count = st.number_input(
            "Number of Passengers",
            min_value=1,
            max_value=6,
            value=1,
            step=1,
            help="Number of passengers in the vehicle (1-6)",
        )

        # Trip distance
        trip_distance = st.number_input(
            "Trip Distance (miles)",
            min_value=0.1,
            max_value=100.0,
            value=5.0,
            step=0.1,
            format="%.1f",
            help="Estimated or actual trip distance",
        )

    with col2:
        # Payment type
        payment_type = st.selectbox(
            "Payment Method",
            options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "Credit Card",
                2: "Cash",
                3: "No Charge",
                4: "Dispute",
            }[x],
            help="How the passenger will pay",
        )

        # Pickup date
        pickup_date = st.date_input(
            "Pickup Date",
            value=datetime(2022, 5, 15),
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2025, 12, 31),
            help="Date of the trip",
        )

        # Pickup time
        pickup_time = st.time_input(
            "Pickup Time", value=time(14, 30), help="Time of pickup"
        )

    st.markdown("---")

    # ============================================
    # PREDICTION SECTION
    # ============================================

    # Predict button (centered and prominent)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "🔮 Predict Fare", use_container_width=True, type="primary"
        )

    # Make prediction when button is clicked
    if predict_button:
        # Combine date and time
        pickup_datetime_str = f"{pickup_date.strftime('%Y-%m-%d')} {pickup_time.strftime('%H:%M:%S')}"

        # Prepare request data
        trip_data = {
            "VendorID": vendor_id,
            "passenger_count": passenger_count,
            "trip_distance": trip_distance,
            "payment_type": payment_type,
            "pickup_datetime": pickup_datetime_str,
        }

        # Show loading spinner
        with st.spinner("🚕 Calculating fare..."):
            result = make_prediction(trip_data)

        # Display results
        if result:
            st.success("✅ Prediction complete!")

            # Main prediction (large and prominent)
            st.markdown("### 💰 Predicted Fare")

            # Display fare in large text
            fare_col1, fare_col2, fare_col3 = st.columns([1, 2, 1])
            with fare_col2:
                st.markdown(
                    f"<h1 style='text-align: center; color: #00cc00;'>"
                    f"${result['predicted_fare']:.2f}"
                    f"</h1>",
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # Additional details in expandable section
            with st.expander("📊 Prediction Details", expanded=True):
                detail_col1, detail_col2 = st.columns(2)

                with detail_col1:
                    st.markdown("**Model Information:**")
                    st.write(f"🤖 Model: {result['model_version']}")
                    st.write(
                        f"🕐 Predicted at: {result['prediction_timestamp']}"
                    )

                with detail_col2:
                    st.markdown("**Engineered Features:**")
                    features = result["input_features"]
                    st.write(f"📅 Hour: {features['pickup_hour']}")
                    st.write(
                        f"📆 Day of week: {features['pickup_day_of_week']}"
                    )
                    st.write(
                        f"📍 Distance: {features['distance_euclidean']:.2f} mi"
                    )

            # Show raw request/response in another expander
            with st.expander("🔍 Technical Details (JSON)", expanded=False):
                st.markdown("**Request:**")
                st.json(trip_data)
                st.markdown("**Response:**")
                st.json(result)

    # ============================================
    # SIDEBAR - INFORMATION & EXAMPLES
    # ============================================

    with st.sidebar:
        st.header("ℹ️ Information")

        st.markdown(
            """
            ### How to use:
            1. Fill in trip details
            2. Click "Predict Fare"
            3. See your estimated fare!
            
            ### About the Model:
            - **Type:** Linear Regression
            - **Dataset:** NYC TLC May 2022
            - **Features:** 8 input features
            
            ### Example Trips:
            """
        )

        # Quick example buttons
        if st.button("📍 Short Trip (2 mi)"):
            st.session_state.example = {
                "distance": 2.0,
                "passengers": 1,
                "payment": 1,
            }

        if st.button("📍 Medium Trip (5 mi)"):
            st.session_state.example = {
                "distance": 5.0,
                "passengers": 2,
                "payment": 1,
            }

        if st.button("📍 Long Trip (15 mi)"):
            st.session_state.example = {
                "distance": 15.0,
                "passengers": 3,
                "payment": 1,
            }

        st.markdown("---")

        st.markdown(
            """
            ### Tech Stack:
            - 🚀 FastAPI
            - 🎨 Streamlit
            - 🤖 Scikit-learn
            - 🐍 Python
            """
        )

        st.markdown("---")
        st.caption("Made with ❤️ for ML Developer Career")


# ============================================
# RUN APP
# ============================================

if __name__ == "__main__":
    main()
