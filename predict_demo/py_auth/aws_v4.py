import sys
import re
import base64
import datetime
import hashlib
import hmac
import requests
import json
import urllib

# This file is derived from the example here
# https://docs.aws.amazon.com/code-samples/latest/catalog/python-signv4-v4-signing-get-post.py.html

def sign(key, msg):
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

  
def getSignatureKey(key, date_stamp, regionName, serviceName):
    kDate = sign(('AWS4' + key).encode('utf-8'), date_stamp)
    kRegion = sign(kDate, regionName)
    kService = sign(kRegion, serviceName)
    kSigning = sign(kService, 'aws4_request')
    return kSigning


def make_query_string(request_dict):
    keys = sorted(request_dict.keys())
    return '&'.join([f'{k}={request_dict[k]}' for k in keys])


def create_v4_headers(access_key, secret_key, method, 
                      path, region, service, host, 
                      request_dict, content_type):
    assert method in ['POST', 'GET']

    # Create a date for headers and the credential string
    t = datetime.datetime.utcnow()
    amz_date = t.strftime('%Y%m%dT%H%M%SZ')
    date_stamp = t.strftime('%Y%m%d') # Date w/o time, used in credential scope

    # ************* TASK 1: CREATE A CANONICAL REQUEST *************
    # http://docs.aws.amazon.com/general/latest/gr/sigv4-create-canonical-request.html

    # Step 1 is to define the verb (GET, POST, etc.)--already done.

    # Step 2: Create canonical URI--the part of the URI from domain to query
    # string (use '/' if no path)
    canonical_uri = path


    # Step 4: Create the canonical headers. Header names must be trimmed
    # and lowercase, and sorted in code point order from low to high.
    # Note that there is a trailing \n.
    if method == 'POST':
        canonical_headers = 'content-type:' + content_type + '\n' + 'host:' + host + '\n' + 'x-amz-date:' + amz_date + '\n'
    elif method == 'GET':
        canonical_headers = 'host:' + host + '\n' + 'x-amz-date:' + amz_date + '\n'

    # Step 5: Create the list of signed headers. This lists the headers
    # in the canonical_headers list, delimited with ";" and in alpha order.
    # Note: The request can include any headers; canonical_headers and
    # signed_headers include those that you want to be included in the
    # hash of the request. "Host" and "x-amz-date" are always required.
    # For DynamoDB, content-type and x-amz-target are also required.
    if method == 'POST':
        signed_headers = 'content-type;host;x-amz-date'
    elif method == 'GET':
        signed_headers = 'host;x-amz-date'

    #
    # Match the algorithm to the hashing algorithm you use, either SHA-1 or
    # SHA-256 (recommended)
    algorithm = 'AWS4-HMAC-SHA256'
    credential_scope = date_stamp + '/' + region + '/' + service + '/' + 'aws4_request'

    if method == 'POST':
        ## Step 3: Create the canonical query string. In this example, request
        # parameters are passed in the body of the request and the query string
        # is blank.
        canonical_querystring = ''

        # Step 6: Create payload hash. In this example, the payload (body of
        # the request) contains the request parameters.

        request_parameters = json.dumps(request_dict)
        payload_hash = hashlib.sha256(request_parameters.encode('utf-8')).hexdigest()
    elif method == 'GET':
        # Step 4: Create the canonical query string. In this example, request
        # parameters are in the query string. Query string values must
        # be URL-encoded (space=%20). The parameters must be sorted by name.
        canonical_querystring = make_query_string(request_dict) # encode to 
#         canonical_querystring += f'&X-Amz-Algorithm={algorithm}'
#         canonical_querystring += '&X-Amz-Credential=' + urllib.parse.quote_plus(access_key + '/' + credential_scope)
#         canonical_querystring += '&X-Amz-Date=' + amz_date
#         canonical_querystring += '&X-Amz-Expires=30'
#         canonical_querystring += '&X-Amz-SignedHeaders=' + signed_headers

        # empty string ("").
        payload_hash = hashlib.sha256(('').encode('utf-8')).hexdigest()
        
    # Step 7: Combine elements to create canonical request
    canonical_request = (method + '\n' + canonical_uri + '\n' 
                         + canonical_querystring + '\n' + canonical_headers + '\n' 
                         + signed_headers + '\n' + payload_hash)
    print('canonical_request, (aka Canonical String)', canonical_request)

    # ************* TASK 2: CREATE THE STRING TO SIGN*************
    string_to_sign = (algorithm + '\n' +  amz_date + '\n' 
                      +  credential_scope + '\n' 
                      +  hashlib.sha256(canonical_request.encode('utf-8')).hexdigest())

    # ************* TASK 3: CALCULATE THE SIGNATURE *************
    # Create the signing key using the function defined above.
    signing_key = getSignatureKey(secret_key, date_stamp, region, service)

    # Sign the string_to_sign using the signing_key
    signature = hmac.new(signing_key, (string_to_sign).encode('utf-8'), hashlib.sha256).hexdigest()


    # ************* TASK 4: ADD SIGNING INFORMATION TO THE REQUEST *************
    # Put the signature information in a header named Authorization.
    authorization_header = algorithm + ' ' + 'Credential=' + access_key + '/' + credential_scope + ', ' +  'SignedHeaders=' + signed_headers + ', ' + 'Signature=' + signature

    return amz_date, authorization_header


def create_api_gateway_auth(url, request_dict, region, access_key, secret_key):
    parts = urllib.parse.urlsplit(url)
    host, path = parts.netloc, parts.path
    method = 'POST'
    service = 'execute-api'
    content_type = 'application/x-amz-json-1.0'

    amz_date, authorization_header = create_v4_headers(
            access_key=access_key,
            secret_key=secret_key,
            method=method,
            path=path,
            region=region,
            service=service,
            host=host,
            request_dict=request_dict,
            content_type=content_type)

    headers = {'Content-Type': content_type,
               'X-Amz-Date': amz_date,
               'Authorization': authorization_header}
    return headers


def GET_create_api_gateway_auth(url, request_dict, region, access_key, secret_key):
    parts = urllib.parse.urlsplit(url)
    host, path = parts.netloc, parts.path
    method = 'GET'

    service = 'execute-api'
    content_type = 'application/x-amz-json-1.0'

    amz_date, authorization_header = create_v4_headers(
            access_key=access_key,
            secret_key=secret_key,
            method=method,
            path=path,
            region=region,
            service=service,
            host=host,
            request_dict=request_dict,
            content_type=content_type)

    headers = {# 'Content-Type': content_type, # XXX hmm
               'X-Amz-Date': amz_date,
               'Authorization': authorization_header}
    return headers


def js_to_python(raw):
    m = re.search(
            (r'\sAccept: "(?P<accept>[^"]*)"\s'
            r'Authorization: "(?P<auth>[^"]*)"\sx-amz-date: "(?P<date>[^"]*)"'), raw); m.groupdict()
    headers = {}

    headers = {
    'Accept': m['accept'],
    'Authorization': m['auth'],
    'x-amz-date': m['date']
    }
    return headers
    # r = requests.get(url=url,
    # headers=headers)
    # return r
