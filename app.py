from flask import Flask, render_template, request
import pandas as pd
import joblib
import folium
import os
import numpy as np

app = Flask(__name__)


MODEL_PATH = "model/rf_hotspot.pkl"
CSV_PATH = "data/train_master.csv"

model = joblib.load(MODEL_PATH)
df = pd.read_csv(CSV_PATH)


density_cols = {
    "June": "jun_density",
    "July": "jul_density",
    "August": "aug_density",
    "September": "sep_density",
    "October": "oct_density",
    "November": "nov_density",
    "December": "dec_density"  
}


feature_cols = [
    "population",
    "land_area_km2",
    "latitude",
    "longitude",
    "aug_density",
    "sep_density",
    "oct_density",
]


severity_colors = {
    "Low": "green",
    "Moderate": "orange",
    "Severe": "red"
}


def classify_severity(density):
    if density < 3000:
        return "Low"
    elif density < 4200:
        return "Moderate"
    else:
        return "Severe"

@app.route("/", methods=["GET", "POST"])
def index():
    map_html = "default_map.html"
    selected_month = "None"

    if request.method == "POST":
        selected_month = request.form["month"]
        X = df[feature_cols].copy()

       
        if selected_month == "December":
            df["dec_predicted_density"] = model.predict(X)

            density_col = "dec_predicted_density"
            severity_col = "dec_predicted_severity"

            df[severity_col] = df[density_col].apply(classify_severity)

        else:
            density_col = density_cols[selected_month]
            severity_col = f"{selected_month.lower()}_severity"

            if severity_col not in df.columns:
                df[severity_col] = df[density_col].apply(classify_severity)

        m = folium.Map(location=[8.482, 124.647], zoom_start=12)

        for _, row in df.iterrows():
            severity = row[severity_col]
            color = severity_colors.get(severity, "gray")

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=8,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"{row['barangay']} - {severity} ({row[density_col]:,.1f} kg/km²)"
            ).add_to(m)

        os.makedirs("static", exist_ok=True)
        m.save("static/map.html")
        map_html = "map.html"

    return render_template(
        "index.html",
        map_html=map_html,
        selected_month=selected_month
    )

if __name__ == "__main__":
    app.run(debug=True)
