# ieds-detection

## Overview
A FastAPI-based image classification API using ResNet50 (Keras/TensorFlow).

## File Structure
```
ieds-detection/
├── Dockerfile                # Docker build instructions for FastAPI app
├── FastAPI.py                # Main FastAPI application
├── README.md                 # Project documentation
├── docker-compose.yml        # Docker Compose configuration
├── lambda_event.py           # AWS Lambda function for SageMaker retraining
├── requirements.txt          # Python dependencies
```
(Images and annotation files for testing are stored in a separate `annotated-images` folder outside this directory.)

## Getting Started

### 1. Clone the Repository
```bash
git clone <your-repo-url>.git
cd ieds-detection
```

### 2. Local Setup (Python)
```bash
pip install -r requirements.txt
uvicorn FastAPI:app --reload
```

### 3. Docker Setup
Build and run the app in Docker:
```bash
docker build -t ieds-detection .
docker run -p 8000:8000 ieds-detection
```

### 4. Docker Compose (Optional)
```bash
docker-compose up --build
```

## API Usage
- `GET /health` — Health check
- `POST /predict/` — Upload an image file (form field: `file`). Returns top-5 predictions.

Example (using curl):
```bash
curl -F "file=@your_image.jpg" http://localhost:8000/predict/
```

### Testing via Web UI (Swagger/OpenAPI)
You can also test the image upload endpoint using the interactive API docs:

1. Open your browser and go to: [http://localhost:8000/docs#/default/predict_predict__post](http://localhost:8000/docs#/default/predict_predict__post)
2. Click the `Try it out` button.
3. For the `file` field, click `Choose File` and select an image from your computer.
4. Click `Execute` to send the request and view the prediction response below.

This web interface is provided by FastAPI and allows you to interactively test all available endpoints.

## Model Retraining with AWS Lambda & SageMaker

Model retraining is managed by the `lambda_event.py` module, which is designed to be deployed as an AWS Lambda function. This function triggers a SageMaker training job using your custom Docker image and S3 data, and now also registers the trained model in the SageMaker Model Registry for versioning and deployment.

### Usage Steps
1. Deploy `lambda_event.py` as an AWS Lambda function (Python 3.x runtime).
2. Set environment variables or pass parameters in the event payload for S3 paths, ECR image, IAM role, etc.
3. Trigger the Lambda function (manually, via S3 event, or on a schedule) to start retraining.
4. The Lambda function will launch a SageMaker training job using the specified Docker image and data.
5. After training, the model artifact will be saved to the specified S3 output path.
6. The Lambda function will then automatically register the trained model in the SageMaker Model Registry, making it available for versioned deployment and approval workflows.

#### Example Lambda Event Payload
```json
{
  "training_image": "<aws_account_id>.dkr.ecr.<region>.amazonaws.com/ieds-detection:latest",
  "role_arn": "arn:aws:iam::<aws_account_id>:role/SageMakerExecutionRole",
  "input_s3": "s3://your-bucket/training-data/",
  "output_s3": "s3://your-bucket/model-artifacts/",
  "instance_type": "ml.m5.large",
  "instance_count": 1,
  "volume_size": 50,
  "max_runtime": 3600,
  "model_package_group_name": "ieds-detection-model-group"
}
```

See `lambda_event.py` for more details and customization options, including model registration in the Model Registry.