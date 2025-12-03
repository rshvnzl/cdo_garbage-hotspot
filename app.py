from flask import Flask, render_template, request
import pandas as pd
import joblib
import folium
import os
import numpy as np

app = Flask(__name__)

# Load model and data
MODEL_PATH = "model/rf_hotspot.pkl"
CSV_PATH = "data/train_master.csv"

model = joblib.load(MODEL_PATH)
df = pd.read_csv(CSV_PATH)

# Density column mapping
density_cols = {
    "June": "jun_density",
    "July": "jul_density",
    "August": "aug_density",
    "September": "sep_density",
    "October": "oct_density",
    "November": "nov_density",
    "December": "dec_density"
}

# Same features used during training!!!
feature_cols = [
    "population",
    "land_area_km2",
    "latitude",
    "longitude",
    "aug_density",
    "sep_density",
    "oct_density",
]


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        selected_month = request.form["month"]

        # ------------------------------
        # 1) Prepare feature matrix X
        # ------------------------------
        X = df[feature_cols].copy()

        # December = predicted month → estimate density
        if selected_month == "December":
            df["dec_density_est"] = df[["aug_density", "sep_density", "oct_density"]].mean(axis=1)
            density_col = "dec_density_est"

        else:
            density_col = density_cols[selected_month]

        # ------------------------------
        # 2) Predict hotspots using RF
        # ------------------------------
        df["predicted_hotspot"] = model.predict(X)

        hotspot_col = "predicted_hotspot"

        # ------------------------------
        # 3) Build the Folium map
        # ------------------------------
        m = folium.Map(location=[8.482, 124.647], zoom_start=12)

        for _, row in df.iterrows():
            color = "red" if row[hotspot_col] == 1 else "green"

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=8,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"{row['barangay']} - "
                      f"{'HOTSPOT' if row[hotspot_col] else 'NORMAL'} "
                      f"({row[density_col]:,.1f} kg/km²)"
            ).add_to(m)

        # Save map file
        os.makedirs("static", exist_ok=True)
        m.save("static/map.html")

        return render_template("index.html", map_html="map.html", selected_month=selected_month)

    # -------- DEFAULT VIEW ON FIRST LOAD --------
    return render_template("index.html", map_html="default_map.html", selected_month="None")


if __name__ == "__main__":
    app.run(debug=True)
