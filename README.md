# 🏆 PM2.5 Air Quality Prediction — Rank #1 Solution

---

## 📌 Competition Overview

The goal was to predict daily **PM2.5 particulate matter concentration** (µg/m³) for 179 cities across the globe — cities that were **completely unseen during training**. This made generalization the central challenge rather than raw predictive power.

Data came from three sources:
- 🛰️ **Sentinel-5P Satellite** — atmospheric pollutants (NO2, CO, HCHO, aerosol index, ozone, etc.)
- 🌦️ **GFS Weather Data** — temperature, humidity, wind components, precipitation
- 📡 **Ground Sensors** — actual PM2.5 readings (train only; target variable)

---

## 🎯 Key Results

| Metric | Score |
|--------|-------|
| Public Leaderboard RMSE | 31.755 |
| Private Leaderboard RMSE | 29.153 |
| Final Rank | **#1 / 93** |

---

## 💡 Winning Insight

The single most impactful decision was the **target transformation**.

Instead of the common `log(1 + y)` transform:

```python
# ❌ log1p — over-compresses extreme values
y_train = np.log1p(train['target'])   # OOF raw RMSE: 27.12

# ✅ sqrt — the winner
y_train = np.sqrt(train['target'])    # OOF raw RMSE: 25.93
```

**Why sqrt won:** The test set contains high-pollution cities the model had never seen. `log1p` compresses large values too aggressively, causing the model to underpredict extreme PM2.5 concentrations. `sqrt` keeps predictions better calibrated across the full range.

---

## 🔧 Solution Pipeline

### 1. Feature Engineering

**Dropped columns**
- All 7 CH4 satellite columns (~81% missing, weak correlation)
- Satellite geometry angles (sensor/solar azimuth & zenith — near-zero correlation)

**Engineered features**

| Feature | Description |
|---------|-------------|
| `wind_speed` | `√(u² + v²)` — actual wind magnitude from GFS components |
| `month`, `week_of_year` | Captured COVID-19 lockdown emission drop from Week 5 |
| `day_of_week`, `day_of_month` | Fine-grained temporal signal |
| `CO_NO2_ratio` | Pollution source type indicator |
| `HCHO_NO2_ratio` | Biomass burning vs traffic signal |
| `CO_HCHO_ratio` | Combustion source ratio |
| Rolling 3-day & 7-day means | Temporal momentum of key pollutants per city |

### 2. City Clustering

Since train and test cities have **zero overlap**, test cities needed an identity. KMeans clustering (k=8) on median satellite + weather profiles grouped all cities into pollution archetypes — then each test city inherited the archetype's historical PM2.5 statistics.

```python
km = KMeans(n_clusters=8, random_state=42, n_init=20)
train_city['city_cluster'] = km.fit_predict(tr_scaled)
test_city['city_cluster']  = km.predict(te_scaled)   # unseen cities get archetype
```

### 3. Model

**LightGBM** with 5-fold cross-validation:

```python
model = LGBMRegressor(
    n_estimators=2000, learning_rate=0.02,
    max_depth=6,       num_leaves=63,
    subsample=0.75,    colsample_bytree=0.75,
    min_child_samples=20,
    reg_alpha=0.1,     reg_lambda=1.0,
    random_state=42
)
```

Predictions reversed with `pred² → PM2.5`.

---

## 📈 Experiment History

| Version | Change | CV RMSE (raw) | LB RMSE |
|---------|--------|---------------|---------|
| Baseline XGBoost | Raw features, log transform | 31.24 | 34.92 |
| v2 — LightGBM | City clusters + ratios + rolling | 27.12 | 32.49 |
| v3 — Over-engineered | Too many features (87 total) | 24.39 | 32.65 ← overfit |
| **Final — sqrt** | Same v2 features, sqrt target | **25.93** | **31.755 🏆** |

The v3 experiment taught a valuable lesson: aggressively improving CV on train cities can *hurt* leaderboard performance when test cities are completely different. Less can be more.

---

## 📂 Repository Structure

```
pm25-air-quality-prediction/
│
├── PM25_Rank1_Solution.ipynb   
├── submission_final.csv        
└── README.md                   
```

> **Note:** `Train.csv`, `Test.csv`, and `SampleSubmission.csv` are not included due to Zindi's data sharing policy. Download them from https://zindi.africa/competitions/university-hackathon-by-olabisi-onabanjo-university/submissions

---

## 🛠️ Requirements

```
pandas
numpy
scikit-learn
lightgbm
matplotlib
```

Install with:
```bash
pip install pandas numpy scikit-learn lightgbm matplotlib
```

---

## 🚀 How to Reproduce

1. Clone the repo and download the data from Zindi
2. Place `Train.csv`, `Test.csv`, `SampleSubmission.csv` in the root directory
3. Open `PM25_Rank1_Solution.ipynb` in Jupyter or Google Colab
4. Run all cells — final predictions saved as `submission_final.csv`

---

## 📚 Data Sources

- **Competition:** [Zindi — University Hackathon by Olabisi Onabanjo University](https://zindi.africa)
- **Weather data:** [NOAA GFS 0.25°](https://developers.google.com/earth-engine/datasets/catalog/NOAA_GFS0P25)
- **Satellite data:** [Copernicus Sentinel-5P via Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/sentinel-5p)



---

*Built as part of ongoing competitive ML practice and community contribution to African AI.*
