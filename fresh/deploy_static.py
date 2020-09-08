
import os
import fresh.s3utils as fs3

def deploy():

    deploy_bucket = os.getenv('S3_DEPLOY_BUCKET')
    assert deploy_bucket
    demo_dir = 'predict_demo'
    for loc in ['index.html', 'js/fetch_data.js']:
        local_loc = f'{demo_dir}/{loc}'
        print(f'copying {local_loc} to {deploy_bucket}:::{loc}')
        with open(local_loc) as fd:
            blah = fd.read()
        s3fn = loc
        fs3.write_s3_file(deploy_bucket, s3fn, blah)

def main():
    pass

