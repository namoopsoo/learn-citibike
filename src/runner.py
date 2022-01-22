import os
import yaml
from glom import glom
from pathlib import Path

import src.aws.cloudformation.utils as sacu
import src.aws.s3 as s3
import src.date_utils as du


def build_sagemaker_stack(dry_run=False):
    knobs = yaml.safe_load(Path("knobs.yaml").read_text())
    bucket = os.getenv("SAGEMAKER_S3_BUCKET")
    aws_account_id = os.getenv("AWS_ACCOUNT_ID")

#    local_loc = "model.tar.gz"
#    s3fn = f"hugging_face/{local_loc}"
#    with open(local_loc, "rb") as fd:
#        blah = fd.read()
#
#    s3.write_this(bucket, s3fn, blah)

    stack_loc = glom(knobs, "Knobs.sagemaker.stack.loc")
    template_body = Path(stack_loc).read_text()

    stack_name = glom(knobs, "Knobs.sagemaker.stack.name")
    endpoint_name = glom(knobs, "Knobs.sagemaker.endpoint.name")
    region = glom(knobs, "Knobs.region")
    model_base_name = glom(knobs, "Knobs.sagemaker.model.docker_image.base_name")
    tag = glom(knobs, "Knobs.sagemaker.model.docker_image.tag")
    s3_loc = glom(knobs, "Knobs.sagemaker.model.s3_loc")
    parameters = {
            "ModelS3Url": 
            f"s3://{bucket}/{s3_loc}",
            "MyEndpointName": endpoint_name,
            "MyModelName": glom(knobs, "Knobs.sagemaker.model.name"),
            
            "DockerImageArn": f"{aws_account_id}.dkr.ecr.{region}.amazonaws.com/{model_base_name}:{tag}"
            }
    print("parameters", parameters)
    if not dry_run:
        output = sacu.manage_stack(stack_name, template_body, parameters)
        print(output)


def s3_lambda_zip_push(lambda_name, deploy_bucket):
    assert deploy_bucket
    local_loc = "foo.zip"
    s3fn = f"{lambda_name}/{local_loc}"
    with open(local_loc, "rb") as fd:
        blah = fd.read()
    s3.write_this(deploy_bucket, s3fn, blah)
    print(f"deployed to {s3fn}")
    return s3fn


def build_lambda_stack(lambda_s3_zip_loc, deploy_bucket):
    with open("src/stacks/LambdaAPIGatewayStack.yaml") as fd:
        template_body = fd.read()
    stack_name = "LambdaAPIGatewayStack"
    sagemaker_endpoint = "ugging-face-one"

    # "s3://them-lambda-artifacts/rearc/code.zip"
    full_lambda_s3_zip_loc = f"s3://{deploy_bucket}/{lambda_s3_zip_loc}"
    parameters = {
        "LambdaS3Loc": full_lambda_s3_zip_loc,
        "SagemakerEndpoint": sagemaker_endpoint,
    }

    print("parameters", parameters)
    output = sacu.manage_stack(stack_name, template_body, parameters)
    print(output)


def build_api_gateway_stack():
    with open("src/stacks/APIGatewayStack.yaml") as fd:
        template_body = fd.read()
    stack_name = "APIGatewayStack"
    parameters = {
        #"": "",
    }
    print("parameters", parameters)
    output = sacu.manage_stack(stack_name, template_body, parameters)
    print(output)
