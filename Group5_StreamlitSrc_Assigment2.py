# Import all the necessary packages
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import streamlit as st
from datetime import datetime, time
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the processed dataset
@st.cache_data
def load_data(filename):
    return pd.read_csv(filename)

# Load the models
def load_model(filename):
    with open(filename, "rb") as file:
        data = pickle.load(file)
        model = data["model"]
        expected_features = data["feature_names"]
        return model, expected_features

# Train-test split function
def train_test_split(df, features, target, testsize=0.2):
    # Select features and target variable
    X = df[features]
    y = df[target]

    # Chronological Train-Test Split
    train_size = int(len(X) * (1 - testsize))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X, y, X_train, X_test, y_train, y_test

# Main function
def main():
    # Container for loading data and models
    with st.container():
        st.markdown("### wLoading Data and Models")
        df = load_data("bike-sharing-hourly.csv")
        df_new = load_data("bike-sharing-hourly-new.csv")
        linreg_model, expected_features = load_model("baseline_linreg_model.pkl")
        xgb_model, expected_features = load_model("best_xgb_model.pkl")

    # Sidebar for user controls
    with st.sidebar:
        st.header("Exploratory Data Analysis Controls")
        hour_filter = st.slider("Select Number of Hours for Display", min_value=1, max_value=len(df), value=17379)
        season_filter = st.selectbox("Select Season", options=["All"] + list(df['season'].unique()))

        # Sidebar for user input
        with st.container():
            st.header("Scenario Inputs for Prediction")
            date_input = st.date_input(
                "Select Date",
                value=datetime(2011, 1, 1).date(),
                min_value=datetime(2011, 1, 1).date(),
                max_value=datetime(2012, 12, 31).date()
            )
            hour_input = st.selectbox("Select Hour", options=list(range(24)), index=12)
            temp_input = st.slider("Temperature (in Celsius)", min_value=0.0, max_value=50.0, value=20.0)
            hum_input = st.slider("Humidity (in %)", min_value=0, max_value=100, value=50)
            windspeed_input = st.slider("Windspeed (in km/h)", min_value=0, max_value=67, value=10)
            weathersit_input = st.slider("Weather Condition", min_value=1, max_value=4, value=2)

    # Combine date and time for prediction
    time_input = time(hour=hour_input)
    datetime_input = datetime.combine(date_input, time_input)

    # Container for prediction
    with st.container():
        st.markdown("### Prediction Based on User Inputs")
        base_row = df_new[df_new["datetime"] == str(datetime_input)]
        if base_row.empty:
            st.error("No data found for the selected date and time.")
        else:
            # Create DataFrame for prediction based on matched row
            user_input_df = base_row.copy()

            # Override user-selected values in DataFrame
            user_input_df["temp"] = temp_input
            user_input_df["hum"] = hum_input
            user_input_df["windspeed"] = windspeed_input
            user_input_df["weathersit"] = weathersit_input

            # Select only features used during model training
            user_input_df = user_input_df[xgb_model.feature_names_in_]

            # Perform prediction
            if st.button("Predict Bike Usage"):
                y_pred = xgb_model.predict(user_input_df)
                st.markdown(
                    f"""
                    <div style="
                        padding: 10px; 
                        border: 2px solid #4682B4; 
                        background-color: #F0F8FF; 
                        color: #4682B4; 
                        border-radius: 5px; 
                        font-size: 18px; 
                        font-weight: bold;
                        text-align: center;
                        ">
                        Predicted Bike Usage: {y_pred[0]:.2f}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # Filter data for Exploratory Data Analysis based on user input
    with st.container():
        st.markdown("### Data Filtering for Exploratory Analysis")
        if season_filter != "All":
            df_s = df_new[df_new[f"season_{season_filter}"] == 1]
        try:
            data_filtered = df_s[-hour_filter:]
        except UnboundLocalError:
            data_filtered = df_new[-hour_filter:]

    # Exploratory Data Analysis Section
    with st.container():
        st.header("Exploratory Data Analysis")
        st.write("This section provides an overview of the bike-sharing data, helping to derive key insights.")

        # Display dataset preview
        st.subheader("Dataset Preview")
        st.write(df.head())

        # Check for missing values
        st.subheader("Missing Values Overview")
        info_data = {
            'Column': df.columns,
            'Non-Null Count': [df[col].notna().sum() for col in df.columns],
            'Dtype': [df[col].dtype for col in df.columns]
        }
        info_df = pd.DataFrame(info_data)
        st.write(info_df)
        st.markdown("It appears that all the data is complete without any missing values.")
        st.markdown("We convert dteday to datetime. Also, we create a new datetime column that combines date and hour for visualizations.")
    
        # Statistics of Variables
        st.subheader("Statistical Summary of Variables")
        st.write(df.describe())
        st.markdown("Please note that windspeed values may need further attention, as maximum values appear higher than expected after normalization.")

        # Distribution plots
        st.subheader("Distribution of Continuous Variables")
        fig, ax = plt.subplots()
        data_filtered[["temp", "atemp", "hum", "windspeed"]].plot(kind='box', ax=ax)
        st.pyplot(fig)
        st.markdown("Temperature and windspeed exhibit some outliers, which may require further consideration.")
        st.markdown("We create columns that track when these values fall in the outlier category.")
        
        # Heatmap of Correlations
        st.subheader("Correlation Analysis")
        fig, ax = plt.subplots(figsize=(10, 6))
        corr = df.drop(columns=["dteday"]).corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)
        st.markdown("Certain features exhibit high correlation, such as temp and atemp. Feature reduction may be required to prevent multicollinearity.")

        # Time Series of Bike Usage Count
        st.subheader("Time Series of Bike Usage Count")
        fig = px.line(data_filtered, x='datetime', y='cnt', title="Bike Usage Over Time")
        st.plotly_chart(fig)

        # Average Bike Usage by Hour
        st.subheader("Average Bike Usage by Hour")
        hourly_usage = data_filtered.groupby('hr')['cnt'].mean().reset_index()
        fig = px.line(hourly_usage, x='hr', y='cnt', title='Average Bike Usage by Hour')
        st.plotly_chart(fig) 
        
        st.subheader("Plot average bike usage by week")
        hourly_usage = data_filtered.groupby('week')['cnt'].mean().reset_index()
        fig = px.line(hourly_usage, x='week', y='cnt', title='Average Bike Usage by Day')
        st.plotly_chart(fig)

        st.subheader("Plot bike usage by season and weather")
        fig = px.box(df, x='season', y='cnt', color='weathersit', title='Bike Usage by Season and Weather')
        st.plotly_chart(fig)
        
        st.markdown("We see that hour cyclicality plays a role and seasonal as well, this visually confirms our suspicions from before. We also see that the heavy rain only happens in the first quarter of the year, perhaps the rain categorical will be redundant")
        
        
        st.subheader("Autocorrelation Analysis")
        st.image("acf.png", caption="ACF")
        st.image("pacf.png", caption="PACF")
        st.markdown("Seems like there is potential predictive power for autoregression")
        st.markdown("Since we're only allowed to use xgboost as a model, we create the following variables which emulate the time series analysis")
        st.markdown("Namely, AR1, AR2, AR3, and MA 1 through 6")

        st.subheader("Cyclicality Analysis")
        st.image("fft.png", caption="Fast Fourrier Transform")
        st.markdown("There are some significant spikes on the chart. We select only the positive ones and only the unique ones up to two digits.")
        st.markdown("We create variables that capture the sin and cos of the cyclicalities fro all the cyclicalities that hte FFT gives us")

        # PART II: Feature Engineering Summary
        st.header("Feature Engineering")
        st.markdown("In addition to all the previously described variables, we also introduce the following to the model:")
        st.write("""
        1. is_clear, is_mist, is_light_rain, is_heavy_rain: Binary columns indicating specific weather situations (clear, mist, light rain, heavy rain), derived from the categorical weathersit variable.

        2. is_peak_hour: Binary indicator for peak hours, set to 1 for high-traffic hours (7-9 AM, 4-7 PM) and 0 otherwise.

        3. temp_bins, hum_bins: Categorical bins for temperature (temp) and humidity (hum), divided into "low," "medium," and "high" ranges based on quantiles.

        4. time_of_day: Categorical indicator of the time of day, divided unevenly to capture rush hours.

        5. season_*, tod_*, temp_bins_*, hum_bins_*: One-hot encoded columns for season, time_of_day, temp_bins, and hum_bins categories.

        6. temp, atemp, hum, windspeed: Denormalized versions of normalized columns to match their original scales.

        7. temp_hum: Interaction feature representing the product of temp and hum, indicating combined effects of temperature and humidity.

        8. temp_windspeed: Interaction feature representing the product of temp and windspeed, indicating combined effects of temperature and wind speed.

        9. weekday_hr: Interaction feature representing the product of weekday and hr, capturing patterns by both day and hour.

        10. casual_registered_ratio: Ratio of casual to registered users, giving insight into user type distribution.

        11. hum_squared: Polynomial feature created by squaring hum, as it has a higher correlation with the target cnt than the original hum value.

        12. daylight: Binary column indicating daylight hours (1 if the hour is between sunrise and sunset, 0 otherwise), based on local sunrise and sunset times for Washington D.C.
                """)

    # Modeling and Predictions
    with st.container():
        st.header("Model Evaluation and Predictions")

        # Display model evaluation for Linear Regression
        st.subheader("Linear Regression Baseline Model")
        X = df_new[expected_features]
        y = df_new["cnt"]    
        X_test = X[int(len(X) * (0.8)):]
        y_test = y[int(len(X) * (0.8)):]
        y_pred_linreg = linreg_model.predict(X_test)
        mae_linreg = mean_absolute_error(y_test, y_pred_linreg)
        rmse_linreg = np.sqrt(mean_squared_error(y_test, y_pred_linreg))
        st.write(f"Mean Absolute Error: {mae_linreg:.2f}")
        st.write(f"Root Mean Squared Error: {rmse_linreg:.2f}")

        # Display model evaluation for XGBoost
        st.subheader("XGBoost Model")
        y_pred_xgb = xgb_model.predict(X_test)
        mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
        rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
        st.write(f"Mean Absolute Error: {mae_xgb:.2f}")
        st.write(f"Root Mean Squared Error: {rmse_xgb:.2f}")

        # Plot Predictions vs Actual
        st.subheader("Model Predictions vs Actual Bike Usage")
        fig, ax = plt.subplots()
        ax.plot(y_test.values, label="Actual", color="blue")
        ax.plot(y_pred_linreg, label="Linear Regression", color="green", alpha=0.6)
        ax.plot(y_pred_xgb, label="XGBoost", color="red", alpha=0.6)
        ax.legend()
        ax.set_title("Predictions vs Actual Bike Usage")
        st.pyplot(fig)
       

    # Recommendations
    with st.container():
        st.header("Recommendations and Key Insights")
        st.markdown("""
                    Overall Pattern: The model captures the overall seasonal and daily fluctuations in bike usage quite well. The peaks and troughs of the predictions closely follow the actual values, indicating that the model is effectively accounting for cyclic patterns in bike demand.

                    Cyclic Trends: The graph shows a clear alignment between predicted and actual values in repeating daily and weekly cycles. This alignment reflects the impact of incorporating Fourier-based cyclical features and lagged values from the SARIMA-inspired configuration (e.g., hourly and multi-hour cyclic terms).

                    Peaks and Valleys: During high-demand periods, the predicted values are very close to the actual values, especially in peak hours. This suggests that the model can accurately forecast busy times, which are critical for the transportation department's planning.

                    Prediction Deviations: While the model tracks trends well, there are slight deviations in some high-fluctuation periods. This is common in time series with irregular, high peaks, where factors not captured by cyclic or lagged features might influence demand.

                    Error Reduction: Compared to the linear model and the XGBoost model without cyclic and lagged features, the residual errors are significantly reduced, as indicated by the tighter clustering of predictions around the actual values.

                    Overall, the graph visually confirms the improved performance of this model, especially in capturing complex, repeating patterns in bike demand, which directly contributes to its lower MAE and RMSE. This strong alignment in the graph demonstrates that the chosen cyclical and lagged features are highly relevant for this problem.
                    """)        
        st.write("""
        Based on the analysis, the following recommendations are suggested:
        - Increase bike availability during peak hours as these times show consistently high demand.
        - Seasonal and weather conditions significantly impact bike usage, and operational strategies should account for these variations.
        - Monitor and adjust service plans on high-windspeed days as bike usage tends to decrease.
        """)
        st.write("Thank you for reviewing the analysis. This interactive dashboard aims to provide data-driven insights to optimize bike-sharing operations in Washington D.C.")

# Run the app
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Bike Sharing Analysis and Prediction")
    st.sidebar.title("Dashboard Options")
    main()