# Mercedes-Benz ADAS Intelligence Platform

A full-stack ML-based ADAS monitoring and development platform for Mercedes-Benz trucks.

## Features
1. **Performance Dashboard** — Real-time metrics, heatmaps, regional compliance
2. **Annotation Tool** — Draw bounding boxes on truck sensor imagery
3. **Model Monitor** — A/B testing, confusion matrix, precision-recall
4. **Incident Analyzer** — Log, cluster, and review ADAS failures
5. **Simulation Hub** — Virtual EU road testing environment
6. **Compliance Dashboard** — ISO 26262, SOTIF, UN R156 tracking
7. **Live Telemetry** — Real-time sensor feeds and ML inference overlay
8. **Explainability** — Grad-CAM saliency maps, feature importance, decision trees

## Deploy to Railway

1. Push this repo to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Select this repo — Railway will auto-detect the Dockerfile
4. Add environment variables:
   - `GROQ_API_KEY` (optional, for AI analysis)
5. Deploy!

The app serves the frontend (`index.html`) and backend API (`/api/*`) on port 5000.

## Local Run

```bash
pip install -r requirements.txt
python backend_api.py
# Open index.html in browser for frontend
```

## Stack
- **Frontend**: Vanilla HTML/CSS/JS (no build step needed)
- **Backend**: Flask + SQLAlchemy
- **Database**: SQLite (dev) / PostgreSQL (prod)
- **AI**: Groq API (optional)
- **Deploy**: Railway / Docker
