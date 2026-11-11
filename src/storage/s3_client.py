
import os

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from pathlib import Path


# Load environment variables from the project root .env file
load_dotenv()

BUCKET_NAME = os.getenv("AWS_S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")


def create_s3_client():
    """Create an authenticated AWS S3 client."""

    if not BUCKET_NAME:
        raise ValueError(
            "AWS_S3_BUCKET is not configured. "
            "Please check your .env file."
        )

    try:
        return boto3.client(
            "s3",
            region_name=AWS_REGION,
        )

    except (BotoCoreError, ClientError) as error:
        raise RuntimeError(
            f"Failed to create S3 client: {error}"
        ) from error


def upload_file(local_file_path, s3_key=None):
    """
    Upload a local file to the configured S3 bucket.

    Args:
        local_file_path: Path to the local file.
        s3_key: Optional destination key in S3.
                Defaults to the local filename.

    Returns:
        The S3 object key.
    """

    file_path = Path(local_file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Path is not a file: {file_path}"
        )

    if s3_key is None:
        s3_key = file_path.name

    if not isinstance(s3_key, str) or not s3_key.strip():
        raise ValueError("S3 object key must be a non-empty string.")

    s3_client = create_s3_client()

    try:
        s3_client.upload_file(
            str(file_path),
            BUCKET_NAME,
            s3_key,
        )

        return s3_key

    except (BotoCoreError, ClientError) as error:
        raise RuntimeError(
            f"Failed to upload {file_path} to S3: {error}"
        ) from error


def download_file(s3_key, local_file_path):
    """
    Download an S3 object to a local file.

    Args:
        s3_key: Source object key in S3.
        local_file_path: Destination path on the local machine.

    Returns:
        Path to the downloaded file.
    """

    if not isinstance(s3_key, str) or not s3_key.strip():
        raise ValueError("S3 object key cannot be empty.")

    file_path = Path(local_file_path)

    if file_path.exists() and file_path.is_dir():
        raise ValueError(
            f"Destination path is a directory: {file_path}"
        )

    file_path.parent.mkdir(parents=True, exist_ok=True)

    s3_client = create_s3_client()

    try:
        s3_client.download_file(
            BUCKET_NAME,
            s3_key,
            str(file_path),
        )

        return file_path

    except ClientError as error:
        error_code = error.response.get(
            "Error", {}
        ).get("Code")

        if error_code in ("404", "NoSuchKey", "NotFound"):
            raise FileNotFoundError(
                f"S3 object not found: {s3_key}"
            ) from error

        raise RuntimeError(
            f"Failed to download {s3_key} from S3: {error}"
        ) from error

    except BotoCoreError as error:
        raise RuntimeError(
            f"Failed to download {s3_key} from S3: {error}"
        ) from error



def object_exists(s3_key):
    """
    Check whether an object exists in the configured S3 bucket.

    Returns:
        True if the object exists, otherwise False.
    """

    if not isinstance(s3_key, str) or not s3_key.strip():
        raise ValueError("S3 object key must be a non-empty string.")

    s3_client = create_s3_client()

    try:
        s3_client.head_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
        )
        return True

    except ClientError as error:
        error_code = error.response.get(
            "Error", {}
        ).get("Code")

        if error_code in ("404", "NoSuchKey", "NotFound"):
            return False

        raise RuntimeError(
            f"Failed to check S3 object: {error}"
        ) from error
