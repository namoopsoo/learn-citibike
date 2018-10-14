

import os
import boto3





def sage_entry(event, context):


    input_dict = event.get('blah')
    what = call_sagemaker(input_dict)


def call_sagemaker(input_dict):

    endpoint = os.getenv('SAGEMAKER_ENDPOINT')

    csvdata = ','.join(input_dict.values())


    client = boto3.client('sagemaker-runtime')


    response = client.invoke_endpoint(
            EndpointName=endpoint,
            # Body=b'bytes'|file,
            Body=csvdata,
            ContentType='txt/csv',
            Accept='string',
            # CustomAttributes='string'
            )
    what = response

    return what

