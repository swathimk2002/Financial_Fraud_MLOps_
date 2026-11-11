
from unittest.mock import MagicMock, patch

import pytest

from src.storage.s3_client import upload_file


def test_upload_valid_file(tmp_path):
    """Verify that a valid file is uploaded to S3."""
    file_path = tmp_path / "transactions.csv"
    file_path.write_text("amount,isFraud\n100,0\n", encoding="utf-8")

    mock_s3 = MagicMock()

    with patch(
        "src.storage.s3_client.create_s3_client",
        return_value=mock_s3,
    ):
        result = upload_file(file_path, "raw/transactions.csv")

    assert result == "raw/transactions.csv"

    mock_s3.upload_file.assert_called_once_with(
        str(file_path),
        "financial-fraud-mlops-swathi-2026",
        "raw/transactions.csv",
    )


def test_upload_file_not_found():
    """Verify that uploading a missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        upload_file("missing_file.csv")


def test_upload_directory_rejected(tmp_path):
    """Verify that directories cannot be uploaded as files."""
    with pytest.raises(ValueError, match="Path is not a file"):
        upload_file(tmp_path)


def test_upload_default_s3_key(tmp_path):
    """Verify that the local filename is used when no S3 key is supplied."""
    file_path = tmp_path / "transactions.csv"
    file_path.write_text("amount\n100\n", encoding="utf-8")

    mock_s3 = MagicMock()

    with patch(
        "src.storage.s3_client.create_s3_client",
        return_value=mock_s3,
    ):
        result = upload_file(file_path)

    assert result == "transactions.csv"

    mock_s3.upload_file.assert_called_once_with(
        str(file_path),
        "financial-fraud-mlops-swathi-2026",
        "transactions.csv",
    )


from src.storage.s3_client import download_file


def test_download_valid_file(tmp_path):
    """Verify that an S3 object is downloaded correctly."""
    destination = tmp_path / "downloads" / "transactions.csv"

    mock_s3 = MagicMock()

    with patch(
        "src.storage.s3_client.create_s3_client",
        return_value=mock_s3,
    ):
        result = download_file(
            "raw/transactions.csv",
            destination,
        )

    assert result == destination
    assert destination.parent.exists()

    mock_s3.download_file.assert_called_once_with(
        "financial-fraud-mlops-swathi-2026",
        "raw/transactions.csv",
        str(destination),
    )


def test_download_empty_s3_key(tmp_path):
    """Verify that an empty S3 key is rejected."""
    with pytest.raises(ValueError, match="S3 object key"):
        download_file("", tmp_path / "file.csv")


def test_download_directory_destination(tmp_path):
    """Verify that an existing directory cannot be used as a file destination."""
    destination = tmp_path / "downloads"
    destination.mkdir()

    with pytest.raises(ValueError, match="Destination path is a directory"):
        download_file("raw/file.csv", destination)


def test_download_missing_s3_object(tmp_path):
    """Verify that a missing S3 object raises FileNotFoundError."""
    from botocore.exceptions import ClientError

    mock_s3 = MagicMock()
    mock_s3.download_file.side_effect = ClientError(
        {
            "Error": {
                "Code": "NoSuchKey",
                "Message": "The specified key does not exist.",
            }
        },
        "GetObject",
    )

    with patch(
        "src.storage.s3_client.create_s3_client",
        return_value=mock_s3,
    ):
        with pytest.raises(FileNotFoundError, match="S3 object not found"):
            download_file(
                "raw/missing.csv",
                tmp_path / "missing.csv",
            )


from src.storage.s3_client import object_exists


def test_object_exists_returns_true():
    """Verify that an existing S3 object returns True."""
    mock_s3 = MagicMock()

    with patch(
        "src.storage.s3_client.create_s3_client",
        return_value=mock_s3,
    ):
        result = object_exists("raw/transactions.csv")

    assert result is True
    mock_s3.head_object.assert_called_once_with(
        Bucket="financial-fraud-mlops-swathi-2026",
        Key="raw/transactions.csv",
    )


def test_object_exists_returns_false():
    """Verify that a missing S3 object returns False."""
    from botocore.exceptions import ClientError

    mock_s3 = MagicMock()
    mock_s3.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}},
        "HeadObject",
    )

    with patch(
        "src.storage.s3_client.create_s3_client",
        return_value=mock_s3,
    ):
        result = object_exists("raw/missing.csv")

    assert result is False


def test_object_exists_rejects_invalid_key():
    """Verify that invalid S3 keys are rejected."""
    with pytest.raises(ValueError, match="non-empty string"):
        object_exists(123)
