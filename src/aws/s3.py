import boto3
from io import StringIO


def read_from_s3(bucket, s3fn, crash_not_found=True): 
    client = boto3.client("s3") 
    try: 
        body = client.get_object(Bucket=bucket, Key=s3fn).get("Body") 
        if body: 
            return body.read() 
    except client.exceptions.NoSuchKey as e: 
        print(f"({bucket}:{s3fn})", repr(e)) 
        if crash_not_found: 
            raise 
        return 


def listdir(bucket, s3prefix):
    o = boto3.client("s3").list_objects_v2(
        Bucket=bucket,
        Prefix=s3prefix,
    )

    contents = o.get("Contents")
    return [] if contents is None else [x["Key"] for x in contents]


def write_this(bucket, s3fn, content):
    client = boto3.client("s3")
    client.put_object(Bucket=bucket, Key=s3fn, Body=content)


def delete_this(bucket, s3fn):
    client = boto3.client("s3")
    client.delete_object(Bucket=bucket, Key=s3fn)
