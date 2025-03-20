import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import math

# Define the sector mapping
sector_mapping = {
    "Agriculture": (0, 5),  # Example mapping; adjust as per your data
    "Education": (5, 8),
    "Health": (8, 10),
    "Export": (10, 12),
    "R&D": (12, 13),
    "Industry": (13, 14),
    "Service": (14, 15),
    "Ease of Doing Business": (15, 20),
    "Inflation Rate": (-10, 10),
}

# Load dataset
@st.cache_data
def load_data():
    try:
        data = pd.read_csv(r"C:\Users\bkred\OneDrive\Desktop\project\Countries.csv")
        return data
    except FileNotFoundError:
        st.error("Data file not found. Please check the file path.")
        st.stop()

# Load the pre-trained models
@st.cache_resource
def load_models():
    try:
        gdp_model = joblib.load("gdp.pkl")  # Regression model for GDP prediction
        investment_model = joblib.load("recommend.pkl")  # Decision tree for investment
        return gdp_model, investment_model
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()

# Initialize data and models
data = load_data()
gdp_model, investment_model = load_models()

# App Title
st.title("Economic Growth Analysis and Prediction Dashboard")

# Sidebar Navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a Page:", ["Home", "Simulations"])

# Home Page
if options == "Home":
    st.header("GDP Analysis and Prediction")
    st.write(
        """This dashboard provides insights into global GDP analysis, sectoral contributions, 
        and simulations. Explore visualizations and simulate changes to GDP by modifying sectoral contributions."""
    )
    st.image("gdp-growth-region-2022-1654625402.jpg")  # Replace with your image path

# Simulations Page
if options == "Simulations":
    st.header("Simulate GDP Changes by Modifying Sectoral Contributions")

    # Dropdown for Continent Selection
    continents = data["Continent Name"].unique()
    continent_selection = st.selectbox("Select a Continent", continents)

    # Filter countries based on selected continent
    countries_in_continent = data[data["Continent Name"] == continent_selection][
        "Country Name"
    ].unique()
    
    # Dropdown to select a country (used for both GDP prediction and global map highlight)
    country_selection = st.selectbox("Select a Country", countries_in_continent)

    if country_selection:
        # Get country data for GDP prediction
        selected_country_data = data[data["Country Name"] == country_selection].iloc[0].to_dict()

        if "Year" in selected_country_data and "GDP" in selected_country_data:
            country_gdp_over_time = data[ 
                data["Country Name"] == country_selection
            ][["Year", "GDP"]]
            fig = px.line(
                country_gdp_over_time,
                x="Year",
                y="GDP",
                title=f"GDP Over Time for {country_selection}",
            )
            st.plotly_chart(fig)
        else:
            st.write("GDP data is not available for the selected country.")

    # Filter the most recent year's data
    recent_year = data["Year"].max()
    recent_data = data[data["Year"] == recent_year]

    # Use the selected country for GDP Prediction and Adjustments
    selected_country_data = recent_data[recent_data["Country Name"] == country_selection].iloc[0].to_dict()

    st.title("GDP Prediction with Feature Adjustments")

    # Sliders for feature adjustments
    sliders = {
        "Agriculture (% GDP)": (0, 70),
        "Education Expenditure (% GDP)": (0, 50),
        "Population Density": (-16, 500),
        "Health Expenditure (% GDP)": (0, 50),
        "Export (% GDP)": (0, 400),
        "R&D (Millions)": (0, 1000),  # Adjusted to display in millions
        "Industry (% GDP)": (0, 90),
        "Service (% GDP)": (0, 100),
        "Ease of Doing Business": (0, 90),
    }

    adjustments = {}
    for feature, (min_val, max_val) in sliders.items():
        default_val = selected_country_data.get(feature, 0)
        if feature == "R&D (Millions)":
            # Adjust default R&D value to millions
            default_val = default_val / 1e6 if default_val and not math.isnan(default_val) else 0
            adjustments[feature] = st.slider(
                feature, min_val, max_val, int(default_val), step=1
            ) * 1e6  # Convert back to original scale
        else:
            if math.isnan(default_val):
                default_val = 0
            adjustments[feature] = st.slider(
                feature, min_val, max_val, int(default_val), step=1
            )

    # Button for GDP Prediction
    if st.button("Predict Simulated GDP"):
        features = [[adjustments[key] for key in adjustments]]
        predicted_gdp = gdp_model.predict(features)[0] / 1e9  # Convert to billion USD
        actual_gdp = selected_country_data["GDP"] / 1e9  # Convert to billion USD

        st.subheader(f"GDP Analysis for {country_selection} ({recent_year}):")
        st.write(f"*Actual GDP (Recent Year):* {actual_gdp:.2f} Billion USD")
        st.write(f"*Predicted GDP (With Adjusted Values):* {predicted_gdp:.2f} Billion USD")

    # Recommend Investment Sectors
    if st.button("Recommend Investment Sectors"):
        investment_features = [[adjustments[key] for key in adjustments]]
        try:
            sector_probabilities = investment_model.predict_proba(investment_features)[0]
        except AttributeError:
            sector_probabilities = investment_model.predict(investment_features).flatten()

        sectors = list(sector_mapping.keys())
        top_3_sectors_indices = sorted(
            range(len(sector_probabilities)),
            key=lambda i: sector_probabilities[i],
            reverse=True,
        )[:3]
        top_3_sectors = [sectors[idx] for idx in top_3_sectors_indices]

        st.subheader("Recommended Sectors to Improve GDP:")
        for idx, sector in enumerate(top_3_sectors, start=1):
            st.write(f"{idx}. {sector}")
   
    # Plot global map
    if "Country Code" in data.columns and "GDP" in data.columns and "Country Name" in data.columns:
        fig = px.choropleth(
            data_frame=data,
            locations="Country Code",  # Use 'Country Code' instead of 'ISO3'
            color="GDP",
            hover_name="Country Name",
            title="Global GDP (Current US$)",
            color_continuous_scale=px.colors.sequential.Viridis,
            template="plotly",
        )

        # Highlight the selected country on the map
        highlight_row = data[data["Country Name"] == country_selection]

        if not highlight_row.empty:
            country_code = highlight_row["Country Code"].values[0]
            fig.add_scattergeo(
                locations=[country_code],
                locationmode="ISO-3",
                text=[country_selection],
                mode="markers+text",
                marker=dict(size=10, color="yellow"),
                name=f"Highlighted: {country_selection}",
            )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Required columns ('Country Code', 'GDP', 'Country Name') are missing in the data.")