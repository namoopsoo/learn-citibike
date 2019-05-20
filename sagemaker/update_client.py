import os
import sys
import boto3
from time import gmtime, strftime


def wrapper_foo(dry_run=True, **kwargs):
    account_id = os.getenv('AWS_ACCOUNT_ID')
    details = {
            'model_data_url': 's3://my-sagemaker-blah/bikelearn/artifacts/citibike-learn-first-job-8-copy-12-04-copy-12-04/20178-12-07-update/model.tar.gz',
            'model_name': kwargs['aws_model_name'],
            'IAM_Role_ARN':
            'arn:aws:iam::{}:role/mySageMakerFullRole'.format(account_id),
            'image':
            '{}.dkr.ecr.us-east-1.amazonaws.com/citibike-learn-blah:{}'.format(
                account_id, kwargs['bike_image_tag']),
            }

    if dry_run:
        return details

    import ipdb ; ipdb.set_trace();
    output = publish_new_model(details)

    print 'done'
    pass


def make_client():
    key_id = os.getenv('MY_ACCESS_KEY_ID')
    secret_key = os.getenv('MY_SECRET_ACCESS_KEY')

    return boto3.client('sagemaker',
            aws_access_key_id=key_id,
            aws_secret_access_key=secret_key,
            region_name='us-east-1')


def publish_new_model(details):
    client = make_client()
    response = client.create_model(
        ModelName=details['model_name'],
        PrimaryContainer={
            # 'ContainerHostname': 'string',
            'Image': details['image'],
            'ModelDataUrl': details['model_data_url'],
            # 'Environment': { 'string': 'string' },
            # 'ModelPackageName': 'string'
        },
        # Containers=[ { 'ContainerHostname': 'string', 'Image': 'string', 'ModelDataUrl': 'string', 'Environment': { 'string': 'string' }, 'ModelPackageName': 'string' }, ],
        ExecutionRoleArn=details['IAM_Role_ARN'],
        # Tags=[ { 'Key': 'string', 'Value': 'string' }, ],
        # VpcConfig={ 'SecurityGroupIds': [ 'string', ], 'Subnets': [ 'string', ] },
        # EnableNetworkIsolation=True|False
    )
    return response


def start_batch_transform_job(**details):
    client = make_client()

    batch_job_name = 'Batch-Transform-' + \
            strftime("%Y-%m-%d-%H%M%S", gmtime())
    print 'batch_job_name ,', batch_job_name 
    model_name = 'pensive-swirles-2-12'
    output_location = 's3://my-sagemaker-blah/bikelearn/datasets/batch_test_sets/'
    input_location = \
            's3://my-sagemaker-blah/bikelearn/datasets/uncompressed/201611-citibike-tripdata.csv'
    # model_name = details['model_name']
    # output_location = details['output_location']
    input_location = details['input_location']

    request = \
            {
                    "TransformJobName": batch_job_name,
                    "ModelName": model_name,
                    "MaxConcurrentTransforms": 4,
                    "MaxPayloadInMB": 8,
                    "BatchStrategy": "SingleRecord",
                    "Environment": {'DO_VALIDATION': 'yes'},
                    "TransformOutput": {
                        "S3OutputPath": output_location,
                        'AssembleWith': 'Line',
                        },
                    "TransformInput": {
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": input_location 
                                }
                            },
                        "ContentType": "text/csv",
                        "SplitType": "Line",
                        "CompressionType": "Gzip"
                        },
                    "TransformResources": {
                        "InstanceType": "ml.m5.large",
                        "InstanceCount": 1
                        }
                    }
    client.create_transform_job(**request)

    print "created transform job with name: ", batch_job_name


def monitor_batch_transform(batch_job_name):
    client = make_client()

    while(True):
        response = client.describe_transform_job(TransformJobName=batch_job_name)
        status = response['TransformJobStatus']
        if  status == 'Completed':
            print("Transform job ended with status: " + status)
            break
        if status == 'Failed':
            message = response['FailureReason']
            print('Transform failed with the following error: {}'.format(message))
            raise Exception('Transform job failed') 
        print("Transform job is still in status: " + status)    
        time.sleep(30)    



if __name__ == '__main__':
    print 'okay ' , sys.argv
    if len(sys.argv) not in [3, 4]:
        sys.exit('bad usage')


    kwargs = {
            'bike_image_tag': sys.argv[1],
            'aws_model_name': sys.argv[2] }
    if len(sys.argv) == 4:
        if sys.argv[3] == 'nodry':
            kwargs.update({'dry_run': False})
        else:
            sys.exit('bad usage. last arg should be "nodry"')

    print wrapper_foo(**kwargs)

