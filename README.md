# Credit Risk Scorecard API

A FastAPI-based credit risk scorecard application that predicts loan default probability and generates credit scores using Weight of Evidence (WOE) transformations and logistic regression.

## 🚀 Features

- **Credit Score Prediction**: Calculate credit scores based on loan application data
- **Risk Assessment**: Determine default probability and risk levels
- **WOE Transformations**: Industry-standard Weight of Evidence feature engineering
- **RESTful API**: Clean, documented API endpoints
- **Web UI**: Simple, responsive web interface
- **Docker Support**: Containerized deployment
- **AWS ECS Ready**: Pre-configured for AWS deployment
- **CI/CD Pipeline**: Automated testing and deployment

## 📁 Project Structure

```
Credit Scorecard - Default Prediction/
├── app/                          # FastAPI application
│   ├── main.py                   # Application entry point
│   ├── models/                   # ML model classes
│   ├── api/                      # API routes and schemas
│   ├── utils/                    # Utility functions
│   └── static/                   # Web UI files
├── data/                         # Data files
├── notebooks/                    # Jupyter notebooks
├── scripts/                      # Training scripts
├── tests/                        # Test files
├── docker/                       # Docker configuration
├── .github/workflows/            # CI/CD pipelines
└── requirements.txt              # Dependencies
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

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run the application**
   ```bash
   python -m app.main
   ```

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose -f docker/docker-compose.yml up --build
   ```

2. **Access the application**
   - API: http://localhost:8000
   - Web UI: http://localhost:8000
   - API Docs: http://localhost:8000/docs

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
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_api.py
```

## 🚀 Deployment

### AWS ECS Deployment

1. **Set up AWS credentials**
   ```bash
   aws configure
   ```

2. **Create ECR repository**
   ```bash
   aws ecr create-repository --repository-name credit-risk-api
   ```

3. **Create ECS cluster**
   ```bash
   aws ecs create-cluster --cluster-name credit-risk-cluster
   ```

4. **Configure GitHub secrets**
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`

5. **Push to main branch to trigger deployment**

### Manual Docker Deployment

1. **Build image**
   ```bash
   docker build -t credit-risk-api -f docker/Dockerfile .
   ```

2. **Run container**
   ```bash
   docker run -p 8000:8000 credit-risk-api
   ```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Environment (development/production) | `development` |
| `DEBUG` | Enable debug mode | `true` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `MODEL_PATH` | Path to trained model | `models/trained_model.pkl` |

### Model Configuration

The model uses the following scoring parameters:
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

## 🔄 Version History

- **v1.0.0**: Initial release with basic credit scoring functionality
- **v1.1.0**: Added batch prediction support
- **v1.2.0**: Enhanced UI and error handling
