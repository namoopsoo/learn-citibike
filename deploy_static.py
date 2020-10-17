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


def api_gateway_hmm():
     apiClient = boto3.client('apigateway', awsregion)
     api_response=apiClient.update_integration
     (
       restApiId=os.getenv('REST_API_ID'),  #,apiName,
       resourceId='myBikelearnSageLambda', # '/api/v1/hub',
       httpMethod='GET',
       integrationHttpMethod='GET',
       type='AWS',
       uri=(f'''arn:aws:lambda:us-east-1:{os.getenv('AWS_ACCOUNT_ID')}:'''
            f'''function:{os.getenv('BIKELEARN_LAMBDA')}'''),
      )


if __name__ == '__main__':
    what = sys.argv[1]
    print('deploying', what)
    if what == 'lambda':
        deploy_lambda()
    elif what == 'html':
        deploy_html()
