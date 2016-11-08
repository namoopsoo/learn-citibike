
import redis
import json

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

def get_classification_results(limit):

    redis_client = connect_redis()

    results = redis_client.lrange(s.CLASSIFICATION_RESULTS_KEY, 0, -1)

    decoded = []

    for result in results:
        decoded.append(json.loads(result))

    return decoded

def _df_from_results(results_list):
    df_results_columns = ['indx','feature_standard_scaling', 'dataset_size',
                          'label_col',
                          'classification', 'accuracy_score', 'f1_score', 'recall_score', 
                          'precision_score']

    results_df = pd.DataFrame(columns=df_results_columns)
    
    for i, row in enumerate(results_list):
        row_dict = deepcopy(row[1])
        row_dict.update(row[2]['test'])
        row_dict.update({'indx': i})
        
        results_df.loc[i] = pd.Series(pd.Series(row_dict))
    return results_df
    
def get_df_from_latest_results(limit=100):

    results = get_classification_results(limit)



