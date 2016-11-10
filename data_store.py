
import redis
import pandas as pd
import json
from copy import deepcopy

import settings as s

def connect_redis():
    '''
from data_store import connect_redis
redis_client = connect_redis()
    '''
    pool = redis.ConnectionPool(host=s.REDIS_SERVER, port=s.REDIS_PORT,
            socket_timeout=s.REDIS_TIMEOUT, db=s.REDIS_DB)
    redis_client = redis.Redis(connection_pool=pool)

    return redis_client


def push_classification_result(result_dict):

    redis_client = connect_redis()

    datajson = json.dumps(result_dict)

    redis_client.lpush(s.CLASSIFICATION_RESULTS_KEY, datajson)

def get_classification_results(limit=-1):
    '''
    when limit is not set, setting limit to -1 will get all the results.
    '''
    redis_client = connect_redis()
    results = redis_client.lrange(s.CLASSIFICATION_RESULTS_KEY, 0, limit)

    decoded = []

    for result in results:
        decoded.append(json.loads(result))

    return decoded

def _df_from_results(results_list):
    df_results_columns = [
            'indx', 'feature_standard_scaling', 'one_hot_encoding',
            'dataset_size', 'label_col', 'classification',
            'training__accuracy_score',
            'test__accuracy_score',
            'holdout__accuracy_score', 'result_date'
            #'f1_score', 'recall_score', 'precision_score'
            ]

    results_df = pd.DataFrame(columns=df_results_columns)
    
    for i, row in enumerate(results_list):
        # [u'definition', u'delta_definition', u'meta', u'results']
        row_dict = deepcopy(row['definition'])
        row_dict.update({'indx': i})
        for type_, result in row['results'].items():
            key_ = '{}__accuracy_score'.format(type_)
            value = result['accuracy_score']
            row_dict.update({key_: value})

        result_meta = result.get('meta', {})
        result_date = result_meta.get('date', '')
        row_dict.update({'result_date': result_date})
        
        results_df.loc[i] = pd.Series(pd.Series(row_dict))
    return results_df
    
def get_df_from_latest_results(limit=100):

    results = get_classification_results(limit)
    df = _df_from_results(results)
    return df



