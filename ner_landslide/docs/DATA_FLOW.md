# Data Flow Architecture

This document explains in beginner-friendly language how data moves through the NER Landslide Early Warning System.

### 1. Data Ingestion (The Senses)
Every 15 minutes, the **Celery Background Worker** (`backend/api/tasks.py`) wakes up and pulls data for each Risk Zone.
- **Open-Meteo**: The system queries the `api.open-meteo.com` endpoint using the exact latitude and longitude of the zone. It retrieves real hourly measurements for `precipitation`, `soil_moisture`, and `weathercode`.
- **USGS Earthquakes**: The system queries `earthquake.usgs.gov` for any recent seismic events occurring near the zone's bounding box.
- **MQTT IoT Simulator**: Ground vibration data is received via MQTT topics. *(Note: This is the ONLY simulated piece of data in the system, due to lack of real public ground sensors).*

### 2. Data Storage (The Memory)
The retrieved JSON data is parsed and saved into our **PostgreSQL (PostGIS)** database using Django Models.
- Measurements are saved as `SensorReading` objects tied to the specific `RiskZone`.
- If an external API fails, the system logs the error and gracefully skips the update (it NEVER invents fake data to fill the gap).

### 3. Risk Engine (The Brain)
Once the new sensor readings are stored, they are passed to the Machine Learning pipeline (`ml/src/predict.py`).
- The ML engine evaluates the combination of soil moisture, heavy rainfall/severity flags, and seismic activity.
- It outputs probabilities for LOW, MEDIUM, and HIGH risk.
- The prediction is saved as a `RiskPrediction` object.

### 4. Alert Engine (The Voice)
If the Risk Engine detects that a zone has escalated to **HIGH** risk:
- An `Alert` object is created in the database.
- A SendGrid email integration (`notify.py`) immediately fires off a warning email to registered authorities.

### 5. React Dashboard (The Face)
Users view the current state of the system via the **React Frontend**.
- The React app fetches data from the Django REST API (`/api/risk-zones/`, `/api/alerts/`).
- The data is displayed on dynamic charts (Recharts) and an interactive map (Leaflet).
- The dashboard highlights data freshness, ensuring users know exactly when the last successful data fetch occurred.
