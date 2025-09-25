# Credit Risk Scorecard for Default Prediction

A **FastAPI-based** credit risk scorecard application that predicts loan default probability and generates credit scores using Weight of Evidence (WOE) transformations and logistic regression.

## 🚀 Features

- **Credit Score Prediction**: Calculate credit scores based on loan application data  
- **Risk Assessment**: Default probability + risk level classification  
- **WOE Transformations**: Industry-standard Weight of Evidence feature engineering  
- **RESTful API**: Clean, documented API endpoints (`/docs` for Swagger UI)  
- **Web UI**: Simple responsive frontend (`index.html`, `app.js`, `style.css`)  
- **Dockerized**: Build and run anywhere with Docker  
- **CI/CD**: GitHub Actions pipeline (linting, type checking, tests, security scan, Docker Hub push)

---

## 📁 Project Structure

```bash
Credit Scorecard - Default Prediction/
├── app/ # FastAPI app
│ ├── main.py # App entrypoint
│ ├── api/ # Routes & schemas
│ ├── models/ # Model logic
│ ├── utils/ # Helper functions
│ └── static/ # Web UI files
├── tests/ # Unit & integration tests
├── Dockerfile # Docker build
├── docker-compose.yml # Local multi-service run
├── requirements.txt # Runtime deps
├── requirements-dev.txt # Dev/test deps
└── .github/workflows/ci.yml # CI/CD pipeline
```

## 🛠️ Installation

### Local Development

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd credit-risk-scorecard
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements-dev.txt
   ```

4. **Run the application**

   ```bash
   uvicorn app.main:app --reload
   ```

### Docker Deployment

1. **Build and run locally**

   ```bash
   docker build -t credit-risk-api .
   docker run -p 8000:8000 credit-risk-api
   ```

   or with Compose:

   ```bash
   docker-compose up --build
   ```

   or pull prebuilt image (from Docker Hub CI/CD)

   ```bash
   docker pull thaitri2005/credit_scorecard_for_default_prediction:latest
   docker run -p 8000:8000 thaitri2005/credit_scorecard_for_default_prediction:latest
   ```

2. **Access the application**
   - API: <http://localhost:8000>
   - Web UI: <http://localhost:8000>
   - API Docs: <http://localhost:8000/docs>

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

## 🧪 Testing

Run the test suite:

```bash
pytest -v
pytest --cov=app --cov-report=html
```

**CI automatically runs:**

- `black` (format check)
- `flake8` (lint)
- `mypy` (type check)
- `pytest` (unit/integration tests)
- `Trivy` (security scan)

## 🔄 CI/CD Workflow

The CI/CD pipeline performs the following steps:

1. **Triggered on every push to `main`**
2. **Runs full test suite**
3. **Builds Docker image**
4. **Pushes Docker image to Docker Hub**  
   Repository: [`thaitri2005/credit_scorecard_for_default_prediction:latest`](https://hub.docker.com/r/thaitri2005/credit_scorecard_for_default_prediction)

For more details, see the [GitHub Actions workflow](.github/workflows/deploy.yml).

### Model Configuration

Logistic regression scorecard with WOE transformations. The model uses the following scoring parameters:

- **PDO (Points to Double Odds)**: 20
- **Base Score**: 600
- **Base Odds**: 50:1

## 📊 Model Features

### Required Features

- `annual_inc`: Annual income
- `int_rate`: Interest rate
- `purpose`: Loan purpose
- `verification_status`: Income verification status

### Optional Features

- `total_rev_hi_lim`: Total revolving high limit
- `tot_cur_bal`: Total current balance
- `loan_burden`: Loan burden ratio
- `revol_util`: Revolving utilization
- `credit_history_length`: Credit history length

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:

- Create an issue on GitHub
- Check the API documentation at `/docs`
- Review the test cases for usage examples
