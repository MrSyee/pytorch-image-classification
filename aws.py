
import os
import argparse

import boto3

from dotenv import load_dotenv


load_dotenv()

s3resource = boto3.resource(
    's3', 
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
)

def save_model_in_s3(model_path: str):
    filename = os.path.basename(model_path)
    s3_bucket = s3resource.Bucket("mlops-study-project")
    s3_bucket.upload_file(model_path, f"dog_cat_classification_models/{filename}")


def parse_args():
    parser = argparse.ArgumentParser(description="Save model in S3.")
    parser.add_argument("model_path", type=str, help="Path of model file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save_model_in_s3(args.model_path)
    print(f"[INFO] Save model in S3: {args.model_path}")