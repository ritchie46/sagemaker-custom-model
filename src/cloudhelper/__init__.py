import io
import boto3


def open_s3_file(bucket, key):
    """
    Open a file from s3 and return it as a file handler.
    :param bucket: (str)
    :param key: (str)
    :return: (stream)
    """
    f = io.BytesIO()
    bucket = boto3.resource('s3').Bucket(bucket)
    bucket.Object(key).download_fileobj(f)
    f.seek(0)
    return f

