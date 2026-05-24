import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium

from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

st.set_page_config(page_title="Cognifyz Restaurant Analysis", layout="wide")

st.title("🍽️ Cognifyz Restaurant Analysis Dashboard")
st.write("Machine Learning Internship Project - 4 Tasks")

df = pd.read_csv("Dataset.csv", encoding="latin1")
df.columns = df.columns.str.replace("ï»¿", "", regex=False)

df = df.dropna(subset=["Cuisines"])

menu = st.sidebar.selectbox(
    "Select Task",
    [
        "Task 1: Rating Prediction",
        "Task 2: Restaurant Recommendation",
        "Task 3: Cuisine Classification",
        "Task 4: Location-Based Analysis"
    ]
)

st.sidebar.write("Cognifyz Technologies Internship")

# ---------------- TASK 1 ----------------
if menu == "Task 1: Rating Prediction":
    st.header("Task 1: Restaurant Rating Prediction")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    data = df[[
        "Average Cost for two",
        "Price range",
        "Votes",
        "Aggregate rating"
    ]].dropna()

    X = data[["Average Cost for two", "Price range", "Votes"]]
    y = data["Aggregate rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    col1, col2 = st.columns(2)
    col1.metric("Mean Squared Error", round(mse, 4))
    col2.metric("R² Score", round(r2, 4))

    st.subheader("Predict Restaurant Rating")

    cost = st.number_input("Average Cost for Two", value=500)
    price = st.selectbox("Price Range", sorted(df["Price range"].dropna().unique()))
    votes = st.number_input("Votes", value=100)

    if st.button("Predict Rating"):
        input_data = pd.DataFrame({
            "Average Cost for two": [cost],
            "Price range": [price],
            "Votes": [votes]
        })

        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Aggregate Rating: {prediction:.2f}")

    st.subheader("Actual vs Predicted Ratings")

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Actual Rating")
    ax.set_ylabel("Predicted Rating")
    ax.set_title("Actual vs Predicted Ratings")
    st.pyplot(fig)

# ---------------- TASK 2 ----------------
elif menu == "Task 2: Restaurant Recommendation":
    st.header("Task 2: Restaurant Recommendation System")

    st.subheader("Select Your Preferences")

    cuisine_options = sorted(df["Cuisines"].dropna().unique())
    city_options = sorted(df["City"].dropna().unique())

    city = st.selectbox("Select City", city_options)
    cuisine = st.selectbox("Select Cuisine", cuisine_options)
    price_range = st.selectbox("Select Price Range", sorted(df["Price range"].dropna().unique()))

    if st.button("Recommend Restaurants"):
        recommendations = df[
            (df["City"] == city) &
            (df["Cuisines"].str.contains(cuisine.split(",")[0], case=False, na=False)) &
            (df["Price range"] == price_range)
        ]

        recommendations = recommendations.sort_values(
            by=["Aggregate rating", "Votes"],
            ascending=False
        )

        st.subheader("Recommended Restaurants")
        st.dataframe(
            recommendations[[
                "Restaurant Name",
                "City",
                "Locality",
                "Cuisines",
                "Average Cost for two",
                "Price range",
                "Aggregate rating",
                "Votes"
            ]].head(10)
        )

# ---------------- TASK 3 ----------------
elif menu == "Task 3: Cuisine Classification":
    st.header("Task 3: Cuisine Classification")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    cuisine_df = df.copy()
    cuisine_df["Main Cuisine"] = cuisine_df["Cuisines"].apply(lambda x: x.split(",")[0])

    top_cuisines = cuisine_df["Main Cuisine"].value_counts().head(10).index
    cuisine_df = cuisine_df[cuisine_df["Main Cuisine"].isin(top_cuisines)]

    data = cuisine_df[[
        "Average Cost for two",
        "Price range",
        "Votes",
        "Aggregate rating",
        "Main Cuisine"
    ]].dropna()

    X = data[["Average Cost for two", "Price range", "Votes", "Aggregate rating"]]
    y = data["Main Cuisine"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.metric("Classification Accuracy", round(accuracy * 100, 2))

    st.subheader("Predict Cuisine")

    cost = st.number_input("Average Cost for Two", value=500)
    price = st.selectbox("Price Range", sorted(df["Price range"].dropna().unique()))
    votes = st.number_input("Votes", value=100)
    rating = st.number_input("Aggregate Rating", value=3.5)

    if st.button("Predict Cuisine"):
        input_data = pd.DataFrame({
            "Average Cost for two": [cost],
            "Price range": [price],
            "Votes": [votes],
            "Aggregate rating": [rating]
        })

        pred = clf.predict(input_data)[0]
        cuisine_name = encoder.inverse_transform([pred])[0]
        st.success(f"Predicted Cuisine: {cuisine_name}")

    st.subheader("Top Cuisine Classes")
    st.dataframe(data["Main Cuisine"].value_counts())

# ---------------- TASK 4 ----------------
elif menu == "Task 4: Location-Based Analysis":
    st.header("Task 4: Location-Based Analysis")

    st.subheader("Restaurant Distribution Map")

    map_df = df.dropna(subset=["Latitude", "Longitude"])
    sample_df = map_df.sample(min(500, len(map_df)), random_state=42)

    m = folium.Map(
        location=[sample_df["Latitude"].mean(), sample_df["Longitude"].mean()],
        zoom_start=3
    )

    for _, row in sample_df.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=3,
            popup=row["Restaurant Name"],
            fill=True
        ).add_to(m)

    st_folium(m, width=900, height=500)

    st.subheader("Top Cities by Number of Restaurants")

    city_count = df["City"].value_counts().head(10)

    fig, ax = plt.subplots()
    city_count.plot(kind="bar", ax=ax)
    ax.set_xlabel("City")
    ax.set_ylabel("Number of Restaurants")
    ax.set_title("Top Cities by Restaurant Count")
    st.pyplot(fig)

    st.subheader("Average Rating by City")

    avg_rating = df.groupby("City")["Aggregate rating"].mean().sort_values(ascending=False).head(10)
    st.dataframe(avg_rating)

    fig2, ax2 = plt.subplots()
    avg_rating.plot(kind="bar", ax=ax2)
    ax2.set_xlabel("City")
    ax2.set_ylabel("Average Rating")
    ax2.set_title("Top Cities by Average Rating")
    st.pyplot(fig2)