"""
lambda_event.py

AWS Lambda function to trigger a SageMaker training job for model retraining and register the trained model in the SageMaker Model Registry.
- Accepts dynamic parameters from the event payload or environment variables.
- Starts a SageMaker training job using a custom Docker image from ECR.
- Waits for training completion and registers the model in the Model Registry.
- Designed for scalable, production-ready ML retraining workflows.
"""

import os
import logging
import boto3
from botocore.exceptions import ClientError
import uuid
from datetime import datetime, timezone
import time

def lambda_handler(event, context):
    """
    Lambda handler to start a SageMaker training job for model retraining and register the model.

    Parameters:
        event (dict): Event payload, can override default/environment config.
        context (LambdaContext): AWS Lambda context object.

    Returns:
        dict: Status code, message, and (on success) SageMaker response.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Use environment variables for config, allow event overrides
    training_image = event.get('training_image', os.getenv('TRAINING_IMAGE', '123456789012.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow:2.4-cpu-py37'))
    role_arn = event.get('role_arn', os.getenv('SAGEMAKER_ROLE_ARN', 'arn:aws:iam::123456789012:role/SageMakerExecutionRole'))
    input_s3 = event.get('input_s3', os.getenv('INPUT_S3', 's3://new-labeled-data/'))
    output_s3 = event.get('output_s3', os.getenv('OUTPUT_S3', 's3://new-trained-models/'))
    instance_type = event.get('instance_type', os.getenv('INSTANCE_TYPE', 'ml.m5.large'))
    instance_count = int(event.get('instance_count', os.getenv('INSTANCE_COUNT', '1')))
    volume_size = int(event.get('volume_size', os.getenv('VOLUME_SIZE', '50')))
    max_runtime = int(event.get('max_runtime', os.getenv('MAX_RUNTIME', '3600')))
    model_package_group_name = event.get('model_package_group_name', os.getenv('MODEL_PACKAGE_GROUP_NAME', 'ieds-detection-model-group'))

    # Generate a unique training job name
    job_suffix = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S') + '-' + str(uuid.uuid4())[:8]
    training_job_name = f"resnet-retraining-{job_suffix}"
    model_name = f"ieds-detection-model-{job_suffix}"

    sagemaker = boto3.client('sagemaker')
    try:
        # Start the training job
        response = sagemaker.create_training_job(
            TrainingJobName=training_job_name,
            AlgorithmSpecification={
                'TrainingImage': training_image,
                'TrainingInputMode': 'File'
            },
            RoleArn=role_arn,
            InputDataConfig=[
                {
                    'ChannelName': 'training',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': input_s3,
                            'S3DataDistributionType': 'FullyReplicated',
                        }
                    }
                }
            ],
            OutputDataConfig={
                'S3OutputPath': output_s3,
            },
            ResourceConfig={
                'InstanceType': instance_type,
                'InstanceCount': instance_count,
                'VolumeSizeInGB': volume_size,
            },
            StoppingCondition={
                'MaxRuntimeInSeconds': max_runtime
            }
        )
        logger.info(f'SageMaker training job {training_job_name} started successfully.')

        # Wait for the training job to complete
        logger.info('Waiting for training job to complete...')
        waiter = sagemaker.get_waiter('training_job_completed_or_stopped')
        waiter.wait(TrainingJobName=training_job_name)
        logger.info('Training job completed.')

        # Get the model artifact S3 path
        desc = sagemaker.describe_training_job(TrainingJobName=training_job_name)
        model_artifact = desc['ModelArtifacts']['S3ModelArtifacts']

        # Create a SageMaker model
        sagemaker.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': training_image,
                'ModelDataUrl': model_artifact
            },
            ExecutionRoleArn=role_arn
        )
        logger.info(f'Model {model_name} created.')

        # Register the model in the Model Registry
        model_package_response = sagemaker.create_model_package(
            ModelPackageGroupName=model_package_group_name,
            ModelPackageDescription=f"Model retrained on {datetime.now(timezone.utc).isoformat()}",
            InferenceSpecification={
                'Containers': [
                    {
                        'Image': training_image,
                        'ModelDataUrl': model_artifact
                    }
                ],
                'SupportedContentTypes': ['application/json'],
                'SupportedResponseMIMETypes': ['application/json']
            },
            ModelApprovalStatus='PendingManualApproval'
        )
        logger.info(f'Model registered in Model Registry: {model_package_response["ModelPackageArn"]}')

        return {
            'statusCode': 200,
            'body': f'Model retraining and registration complete: {training_job_name}',
            'training_job_name': training_job_name,
            'model_name': model_name,
            'model_package_arn': model_package_response["ModelPackageArn"]
        }
    except ClientError as e:
        logger.error(f"SageMaker operation failed: {e}")
        return {
            'statusCode': 500,
            'body': f'Error: {e}'
        }
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {
            'statusCode': 500,
            'body': f'Unexpected error: {e}'
        }
