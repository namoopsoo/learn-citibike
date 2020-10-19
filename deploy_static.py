import boto3
import os
import sys
import fresh.s3utils as fs3
import fresh.utils as fu


def deploy_html():
    deployables = ['index.html', 'js/fetch_data.js']

    deploy_bucket = os.getenv('S3_DEPLOY_BUCKET')
    assert deploy_bucket
    demo_dir = 'predict_demo'
    for loc in deployables:
        local_loc = f'{demo_dir}/{loc}'
        print(f'copying {local_loc} to {deploy_bucket}:::{loc}')
        with open(local_loc) as fd:
            blah = fd.read()
        s3fn = loc
        if loc.endswith('html'):
            content_type = 'text/html'
        else:
            content_type = None
        fs3.write_s3_file(deploy_bucket, s3fn, blah, content_type=content_type)


def read_git_hash():
    with open('fresh/git_hash.txt') as fd: 
        return fd.read().split()[0]


def deploy_lambda():
    s3loc = s3_lambda_zip_push()
    client = boto3.client('lambda',
            region_name='us-east-1'
            )
    out = client.update_function_code(
        FunctionName=os.getenv('BIKELEARN_LAMBDA'),
        # ZipFile=b'bytes',
        S3Bucket=os.getenv('S3_LAMBDA_ARTIFACTS_BUCKET'),
        S3Key=s3loc,
        # S3ObjectVersion='string',
        Publish=True,
        DryRun=False,
        # RevisionId=last_revision_id
    )
    print(fu.subset(out, ['Version', 'LastModified', 'LastUpdateStatus', 'State']))
    return out


def s3_lambda_zip_push():
    deploy_bucket = os.getenv('S3_LAMBDA_ARTIFACTS_BUCKET')
    assert deploy_bucket
    local_loc = 'foo.zip'
    s3fn = f'{os.getenv("BIKELEARN_LAMBDA")}/{read_git_hash()}/{local_loc}'
    with open(local_loc, 'rb') as fd:
        blah = fd.read()
    fs3.write_s3_file(deploy_bucket, s3fn, blah)
    print(f'deployed to {s3fn}')    
    return s3fn


def api_gateway_hmm_update_lambda_version(version):
    client = boto3.client('apigateway', 
                           region_name='us-east-1')
    patch_uri = [{
        'path': '/uri',
        'value': f'''arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:{os.getenv('AWS_ACCOUNT_ID')}:function:myBikelearnSageLambda:{version}/invocations''',
        'op': 'replace'
        }]
    print('patch_uri', patch_uri)
    response = client.update_integration(
             restApiId=os.getenv('REST_API_ID'),
             resourceId='tsgjxn',
             httpMethod='GET',
             patchOperations=patch_uri,
             )
    return response


def add_permission(version):
    '''
    $ aws lambda add-permission --function-name LambdaFunctionOverHttps \
--statement-id apigateway-test-2 --action lambda:InvokeFunction \
--principal apigateway.amazonaws.com \
--source-arn "arn:aws:execute-api:$REGION:$ACCOUNT:$API/*/POST/DynamoDBManager"
{
    "Statement": "{\"Sid\":\"apigateway-test-2\",\"Effect\":\"Allow\",\"Principal\":{\"Service\":\"apigateway.amazonaws.com\"},\"Action\":\"lambda:InvokeFunction\",\"Resource\":\"arn:aws:lambda:us-east-2:123456789012:function:LambdaFunctionOverHttps\",\"Condition\":{\"ArnLike\":{\"AWS:SourceArn\":\"arn:aws:execute-api:us-east-2:123456789012:mnh1yprki7/*/POST/DynamoDBManager\"}}}"
}
    '''
    region = 'us-east-1'
    source_arn = (f"arn:aws:execute-api:{region}:{os.getenv('AWS_ACCOUNT_ID')}"
                   f":{os.getenv('REST_API_ID')}/default/GET/myBikelearnSageLambda")
    print('DEBUG', source_arn)
    client = boto3.client('lambda', 
                           region_name='us-east-1')
    response = client.add_permission(
        FunctionName=os.getenv("BIKELEARN_LAMBDA"),
        StatementId='blah-statement-2', # TODO randomize
        Action='lambda:InvokeFunction',
        Principal='apigateway.amazonaws.com',
        SourceArn=source_arn,
        SourceAccount=os.getenv('AWS_ACCOUNT_ID'),
        # EventSourceToken='string', # NOTE Alexa?
        Qualifier=f'{version}',
        # RevisionId='string'
    )
    return response



def deploy_api(description):
    client = boto3.client('apigateway', 
                           region_name='us-east-1')
    response = client.create_deployment(
            restApiId=os.getenv('REST_API_ID'),
            stageName='default',
            # stageDescription='string',
            description=description,
            )
    return response

if __name__ == '__main__':
    what = sys.argv[1]
    print('deploying', what)
    if what == 'lambda':
        deploy_lambda()
    elif what == 'html':
        deploy_html()
