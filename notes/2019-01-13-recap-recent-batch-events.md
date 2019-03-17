#### Quick recap
* Recently I have been trying to batch-run the model trained on an early month in ~2016, against several months of data, to try to observe performance degradation.
* First run of 201603 failed, with error , 
```
2018/12/09 21:38:02 [error] 9#9: *4 client intended to send too large body: 6291424 bytes, client: 169.254.255.130, server: , request: "POST /invocations HTTP/1.1", host: "169.254.255.131:8080"
```
* Then after increasing the Docker  `nginx.conf`  from  `client_max_body_size 5m;` to `client_max_body_size 8m;`  , trying `201603` again worked. 
* My job on `201609` failed , `pensive-swirles-job-201609` , because of the error `ClientError: S3 key: s3://my-sagemaker-blah/bikelearn/datasets/uncompressed/201609-citibike-tripdata.csv matched no files on s3` , so that one was a simple fix , 
* Subsequently, the error on `pensive-swirles-job-201610` , `201611` and beyond failed with the error , 
```
169.254.255.130 - - [24/Dec/2018:04:20:48 +0000] "POST /invocations HTTP/1.1" 500 291 "-" "Go-http-client/1.1"
ValueError: time data '2016-10-01 00:00:07' does not match format '%m/%d/%Y %H:%M:%S'

```
* Because the file formats on `201610`, `201611`, `201612` as far as i can tell have changed the date formats. 
* I corrected the Docker code reading dates from `pensive-swirles-2-11` to `pensive-swirles-2-12` and the problem went away.

#### But furthermore...
* I then automated the batch transform w/ the API , but the first time around job `Batch-Transform-2019-01-06-233207` failed because i forgot to include my special ENVIRONMENT variable, `DO_VALIDATION` for indicating  validation. 
* I did add this for the job `Batch-Transform-2019-01-06-235825` and that one was successful.

#### but then the log ...
* for `Batch-Transform-2019-01-06-235833` , has a mix bag of success and also timeouts, 
```
2019/01/07 00:02:10 [error] 9#9: *10 upstream timed out (110: Connection timed out) while sending request to upstream, client: 169.254.255.130, server: , request: "POST /invocations HTTP/1.1", upstream: "http://unix:/tmp/gunicorn.sock/invocations", host: "169.254.255.131:8080"
```
* And i dont see the typical output file generated for this on s3, so i dont think this finished.

