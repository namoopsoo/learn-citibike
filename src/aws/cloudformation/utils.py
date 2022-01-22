import boto3
import yaml


def validate_stack_template(template):

    client = boto3.client("cloudformation", region_name="us-east-1")
    return client.validate_template(
        TemplateBody=template
    )

def manage_stack(stack_name, template_body, parameters):
    print(validate_stack_template(template_body))
    print("validation result ^^")

    client = boto3.client("cloudformation", region_name="us-east-1")

    response = client.create_stack(StackName=stack_name,
            TemplateBody=template_body,
            Capabilities=[
                "CAPABILITY_AUTO_EXPAND",
                "CAPABILITY_IAM"
            ],
            Parameters=[
                {"ParameterKey": k, "ParameterValue": v} 
                for (k, v) in parameters.items()
            ],

            )
    return response
