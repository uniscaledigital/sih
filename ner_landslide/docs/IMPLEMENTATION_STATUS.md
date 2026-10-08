# Implementation Status

| Feature | Status | Notes |
| :--- | :--- | :--- |
| **Open-Meteo live API** | [✓] | Pulling precipitation, soil moisture, and severity codes. |
| **USGS live API** | [✓] | Polling bounding box coordinates. |
| **Terrain data (DEM)** | [~] | Backend Model created; awaiting massive DEM GIS file ingestion. |
| **GSI inventory** | [ ] | Awaiting labelled dataset for ML training. |
| **ML training** | [ ] | Scripts written (`fetch_training_data`, `preprocess`, `train`), awaiting real labels. |
| **SHAP explainability** | [~] | Backend storage & UI widget ready; awaiting real trained model. |
| **PostgreSQL** | [✓] | Storing measurements, alerts, and zones. |
| **PostGIS** | [✓] | Handling Polygon and Point geometric fields. |
| **Celery** | [✓] | Processing async data pulls every 15 minutes. |
| **Redis** | [✓] | Managing Celery message queues. |
| **React dashboard** | [✓] | Fetching real database values; map active. |
| **Alerts** | [✓] | SendGrid integration actively fires on HIGH risk. |
