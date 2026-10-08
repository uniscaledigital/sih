# Beginner Setup Guide

Welcome to the NER Landslide Early Warning System prototype! This guide will walk you through exactly how to set up and run the system from scratch.

## Prerequisites
- Docker & Docker Compose
- Python 3.10+
- Node.js (for frontend local development, though Docker handles the build)
- Git

## Step-by-Step Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/ner-landslide-ews.git
cd ner-landslide-ews
```

### 2. Environment Configuration
Copy the `.env.example` to `.env` in the `backend/` directory:
```bash
cp backend/.env.example backend/.env
```
Ensure you provide your SendGrid API key if you want to test email alerts. Open-Meteo and USGS do not require API keys.

### 3. Start the Docker Services
We use Docker Compose to orchestrate PostgreSQL, Redis, Django, and Celery.
```bash
docker-compose up -d --build
```

### 4. Database Migration & Seeding
Once the containers are running, apply the database migrations and seed the initial Kohima district zones:
```bash
docker-compose exec backend python manage.py migrate
docker-compose exec backend python manage.py seed_mock_data
```

### 5. Start the React Frontend
Open a new terminal and navigate to the frontend folder:
```bash
cd frontend
npm install
npm run dev
```
The dashboard will be available at `http://localhost:5173`.

### 6. Verify API Connections
You can test the system's live connections to Open-Meteo and USGS via the Health API:
```bash
curl http://localhost:8000/api/health/
```

To view the latest data source fetching status:
```bash
curl http://localhost:8000/api/data-sources/status/
```

### 7. Trigger a Manual Risk Recalculation
Celery runs every 15 minutes automatically, but you can force an immediate refresh of weather and seismic data:
```bash
docker-compose exec backend python manage.py shell -c "from api.tasks import recalculate_risk; recalculate_risk()"
```

### 8. MQTT Simulator (Optional)
To run the simulated ground vibration IoT sensor:
```bash
cd iot-simulator
python mqtt_publisher.py
```
*(Note: This is the ONLY component in this project that uses simulated data).*
