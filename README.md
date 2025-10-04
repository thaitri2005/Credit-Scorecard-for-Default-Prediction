# 💳 Credit Risk Scorecard for Default Prediction

A **FastAPI-based web application** for **credit default risk prediction**, adapted from my Kaggle notebook  
👉 [Credit Risk Scorecard for Default Prediction](https://www.kaggle.com/code/quangnguynngnht/credit-risk-scorecard-for-default-prediction).

The notebook covers the **entire modeling pipeline**, while this repository turns that work into a **production-ready, Dockerized API + Web UI** with **CI/CD automation** and **Azure Cloud deployment**.

---

## 🚀 Live Deployment

**🌐 Production App (Azure):**  
🔗 [https://creditscorecardwoeiv-bgc3edb0fzgzdzfg.eastasia-01.azurewebsites.net/](https://creditscorecardwoeiv-bgc3edb0fzgzdzfg.eastasia-01.azurewebsites.net/)

> ⚙️ *Startup may take 30–60 seconds on first access (free tier cold start).*

---

## 🚀 Features

- **Credit Score Prediction** using a Logistic Regression model with **Weight of Evidence (WOE)** transformation  
- **Default Risk Assessment**: Predicts probability + credit category  
- **Preprocessing Pipeline** (feature binning, scaling, encoding)  
- **RESTful API** built with FastAPI  
- **Interactive Swagger Docs** at `/docs`  
- **Responsive Web UI** for non-technical users  
- **Dockerized Deployment** via Azure App Service or Docker Compose  
- **Full CI/CD Workflow** using GitHub Actions (lint → test → build → push → deploy)

---

## 📁 Project Structure

```bash
Credit Scorecard - Default Prediction/
├── app/                      # FastAPI application
│   ├── main.py               # App entrypoint
│   ├── api/                  # API layer: routes & schemas
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── utils/                # Helper utilities
│   │   └── helpers.py
│   └── static/               # Web UI (HTML, CSS, JS, assets)
├── tests/                    # Unit & integration tests
├── models/                   # Exported pipeline, model, scorecard, metadata
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
└── .github/
    └── workflows/
        └── deploy.yml            # CI/CD pipeline
```

---

📚 Background

This project is based on the Kaggle notebook:
👉 [Credit Risk Scorecard for Default Prediction](https://www.kaggle.com/code/quangnguynngnht/credit-risk-scorecard-for-default-prediction)

The data preparation, WOE binning, and logistic regression model training are all explained in the notebook.

The exported preprocessing pipeline, trained model, and scorecard are taken from that notebook.

This repository focuses on wrapping the trained pipeline + model into a FastAPI application, with a frontend and Dockerized CI/CD pipeline for real-world usage.

---

## 🛠️ Installation

### Local Development

```bash
git clone https://github.com/thaitri2005/Credit-Scorecard-for-Default-Prediction.git
cd credit-risk-scorecard
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload

```

### Docker Deployment

   ```bash
   docker build -t credit-risk-api .
   docker run -p 8000:80 credit-risk-api
   ```

   or with Compose:

   ```bash
   docker-compose up --build
   ```

   or pull prebuilt image (from Docker Hub CI/CD)

   ```bash
   docker pull thaitri2005/credit_scorecard_for_default_prediction:latest
   docker run -p 8000:80 thaitri2005/credit_scorecard_for_default_prediction:latest
   ```

---

## 📚 API Documentation

### Endpoints

- `GET /` - Web UI
- `GET /api/v1/health` - Health check
- `POST /api/v1/predict` - Single prediction
- `POST /api/v1/predict/batch` - Batch predictions
- `GET /api/v1/model/info` - Model information
- `GET /api/v1/model/features/importance` - Feature importance
- `GET /docs` - Interactive API documentation

### Example Request

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "annual_inc": 75000,
       "int_rate": 12.5,
       "credit_history_length": 5.5,
       "purpose": "debt_consolidation",
       "verification_status": "verified"
     }'
```

### Example Response

```json
{
  "credit_score": 650.25,
  "default_probability": 0.1234,
  "risk_level": "Medium Risk",
  "log_odds": -1.9876,
  "message": "Prediction completed successfully"
}
```

---

## 🔄 CI/CD Workflow

The CI/CD pipeline performs the following steps:

1. Lint & Type Check
2. Run Unit Tests
3. Security Scan (Trivy)
4. Build Docker Image
5. Push to Azure Container Registry
6. Deploy to Azure Web App (App Service)

For more details, see the [GitHub Actions workflow](.github/workflows/deploy.yml).

---

## 📊 Model Features

- `annual_inc`: Annual income
- `int_rate`: Interest rate
- `purpose`: Loan purpose
- `verification_status`: Income verification status
- `total_rev_hi_lim`: Total revolving high limit
- `tot_cur_bal`: Total current balance
- `loan_burden`: Loan burden ratio
- `revol_util`: Revolving utilization
- `credit_history_length`: Credit history length

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🆘 Support

For support and questions:

- Create an issue on GitHub
- Check the API documentation at `/docs`
- Review the test cases for usage examples
