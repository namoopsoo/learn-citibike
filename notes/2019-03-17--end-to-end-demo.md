#### Okay , get my project demo-ready!
What is left
* 

#### Can I hit the endpoint, what does it give?
* So my raw API Gateway endpoint is
https://rmuxqpksz2.execute-api.us-east-1.amazonaws.com/default/myBikelearnSageLambda?start_station=Forsyth+St+%26+Broome+St&start_time=10%2F8%2F2015+18%3A04%3A57&rider_gender=2&rider_type=Subscriber&birth_year=1973
* Currently outputs 
```{"output": "17\n"}```
* Can I create a quick page around this to let someone use it and interpret/receive the output nicely?
* And of course, I want input examples that produce different outputs. And hey what about the ranked output also.

#### Mini s3 index html page that hits this endpoint
* Will use my other page for inspiration! http://time-parse.s3-website-us-east-1.amazonaws.com/stacked.html?start=2017-12-01&end=2017-12-22    

```

https://console.aws.amazon.com/s3/buckets/bike-hop-predict/?region=us-east-1&tab=permissions

http://bike-hop-predict.s3-website-us-east-1.amazonaws.com/index.html

```
