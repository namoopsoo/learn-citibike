import os
import boto3
import botocore


def sage_entry(event, context):
    '''
    input_dict = {
            "start_time": "10/1/2015 00:00:02",
            "start_station": "W 26 St & 10 Ave",
            "rider_type": "Subscriber",
            "birth_year": "1973",
            "rider_gender": "1"
            }
    '''
    input_dict = event.get('blah')


    what = call_sagemaker(input_dict)
    return what


def call_sagemaker(input_dict):

    endpoint = os.getenv('SAGEMAKER_ENDPOINT')
    key_id = os.getenv('MY_ACCESS_KEY_ID')
    secret_key = os.getenv('MY_SECRET_ACCESS_KEY')

    csvdata = ','.join(input_dict.values())

    csvdata = "10/1/2015 00:00:02,W 26 St & 10 Ave,Subscriber,1973,1"

    client = boto3.client('sagemaker-runtime',
            aws_access_key_id=key_id,
            aws_secret_access_key=secret_key,
            region_name='us-east-1')

    try:
        response = client.invoke_endpoint(
                EndpointName=endpoint,
                # Body=b'bytes'|file,
                Body=csvdata,
                ContentType='text/csv',
                Accept='string',
                # CustomAttributes='string'
                )
        what = response['Body'].read()
        return {'output': what}
    except botocore.exceptions.ClientError as e:
        error = {'error_detail': str(e.message), 'error': 'client error'}
        return error


def call_sagemaker_with_session_quick_test():

    session = boto3.Session(profile_name='adminuser')

    client = session.client('sagemaker-runtime')
    endpoint = "bikelearn-astral-ankle"

    csvdata = "10/1/2015 00:00:02,W 26 St & 10 Ave,Subscriber,1973,1"

    response = client.invoke_endpoint(
            EndpointName=endpoint,
            # Body=b'bytes'|file,
            Body=csvdata,
            ContentType='text/csv',
            Accept='string',
            # CustomAttributes='string'
            )
    what = response

    return what



