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

# Map month names to density column names
density_cols = {
    "June": "jun_density",
    "July": "jul_density",
    "August": "aug_density",
    "September": "sep_density",
    "October": "oct_density",
    "November": "nov_density",
    "December": "dec_density"
}

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        selected_month = request.form["month"]

        if selected_month == "December":

            feature_cols = ["population","land_area_km2","latitude","longitude",
                            "aug_density","sep_density","oct_density"]

            X_dec = df[feature_cols].copy()
            X_dec["oct_density"] = df["oct_density"]

            df["dec_predicted_hotspot"] = model.predict(X_dec)
            df["dec_density_est"] = df[["aug_density", "sep_density", "oct_density"]].mean(axis=1)

            hotspot_col = "dec_predicted_hotspot"
            density_col = "dec_density_est"

        else:
            density_col = density_cols[selected_month]
            threshold = 4300
            hotspot_col = f"{selected_month.lower()}_is_hotspot"

            if hotspot_col not in df.columns:
                df[hotspot_col] = (df[density_col] > threshold).astype(int)

        # Create Folium map
        m = folium.Map(location=[8.482, 124.647], zoom_start=12)
        for _, row in df.iterrows():
            color = "red" if row[hotspot_col] == 1 else "green"
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=8,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"{row['barangay']} - {'HOTSPOT' if row[hotspot_col] else 'NORMAL'} ({row[density_col]:.1f} kg/km²)"
            ).add_to(m)

        os.makedirs("static", exist_ok=True)
        m.save("static/map.html")

        map_html = "map.html"

    else:
        # --- DEFAULT MAP ON FIRST LOAD ---
        selected_month = "None"
        map_html = "default_map.html"   # ONLY filename, no "static/"

    return render_template("index.html", map_html=map_html, selected_month=selected_month)

if __name__ == "__main__":
    app.run(debug=True)
