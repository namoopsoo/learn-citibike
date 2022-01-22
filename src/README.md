

#### Quick prerequisites
* Make sure you have a python environment with `awscli`
* Make sure you have run `aws configure` too, to update your `~/.aws/config` . 

#### Activate the sagemaker stack


First set a few env variables

```bash
export AWS_ACCOUNT_ID=
export SAGEMAKER_S3_BUCKET=
```

Then can run, from a shell. 

```python
from src import runner
runner.build_sagemaker_stack()

```
