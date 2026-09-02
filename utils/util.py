import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import hashlib
import os
import requests
import time

QUERY_CACHE_DIR = "cache"
CACHE_EXT = ".json"

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def send_data(url: str, data: list, try_cnt: int = 3):
    cnt = 0
    while cnt < try_cnt:
        try:
            res = requests.post(url=url, data=json.dumps(data), timeout=10)
            print(f"Send Success : {res.status_code}")
            return res.status_code
        except requests.exceptions.ConnectTimeout as ct:
            print(f"Connect Timeout, Send Retry : {ct}")
            cnt += 1
            time.sleep(1)
        except requests.exceptions.ConnectionError as ce:
            print(f"Connection error, Send Retry : {ce}")
            cnt += 1
            time.sleep(1)
        except requests.exceptions.ReadTimeout as rte:
            print(f"Read Time error, Send Retry : {rte}")
            cnt += 1
            time.sleep(1)

def send_analysis_data(data: list, url):
    for i in range(0, len(data), 100):
        send_data(url, data[i : i + 100])
        time.sleep(0.5)

def get_analysis_data(
    metric: str,
    timestamp: int,
    value: float,
    tag: str,
    ):
    tags=tag.split('.')
    for i in range(len(tags), 4, 1):
        tags.append('_')

    mydata = {
        "metric": metric,
        "timestamp": timestamp,
        "value": value,
        "tags": {"place": tags[0], "typeno": tags[1], "feeder": tags[2],"tag":tags[3]}
    }
    return mydata

def post_data(query, data, tag_name):   
    #send data
    metric=query['metric']
    place=query['place']
    typeno=query['typeno']
    feeder=query['feeder']
    tag=query['tag']
    start=query['date']['start']
    end=query['date']['end']
    port=query['port']
    url=query['url']
    api=query['api']
    df = data

    pre = Preprocessing(metric, place, typeno, feeder, tag, start, end, port, url, api)
    df = pre.push_data(df)
    #send data
    timestamp = [list(d['dps'].keys()) for d in df['results']][0]
    name = f"{place}.{typeno}.{feeder}.{tag}_{tag_name}"
    value = [list(d['dps'].values()) for d in df['results']][0]

    my_data_lst = []
    for times, val in zip(timestamp, value):
        my_data_lst.append(get_analysis_data(metric, times, val, name))

    #send_analysis_data(my_data_lst, url)

def get_data(
    url: str, using_cache: bool = False, print_cache: bool = False, try_cnt: int = 5
):
    if not os.path.exists("./cache"):
        os.mkdir("./cache")
    hash = hashlib.sha256(url.encode()).hexdigest()
    path = f"{QUERY_CACHE_DIR}/{hash}.{CACHE_EXT}"
    if print_cache:
        print(hash)

    # 파일이 이미 존재?
    if using_cache and os.path.isfile(path):
        with open(path, "r") as f:
            data = json.load(f)
        return data

    # 파일이 없다면 데이터 받아오기

    cnt = 0
    while cnt < try_cnt:
        try:
            r = requests.get(url, timeout=10)
            data = r.json() if json else r.text
            print(f"Get Success : {r.status_code}")

            if using_cache:
                with open(path, "w") as f:
                    json.dump(data, f)
            return data

        except requests.exceptions.ConnectTimeout as e:
            print(f"Get Retry : {e}")
            cnt += 1
            time.sleep(1)

    print("Get Fail")
    return None




class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

def load_data(query):
    #get data
    metric=query['metric']
    place=query['place']
    typeno=query['typeno']
    feeder=query['feeder']
    tag=query['tag']
    start=query['date']['start']
    end=query['date']['end']
    port=query['port']
    url=query['url']
    api=query['api']
    save=query['save']


    pre = Preprocessing(metric, place, typeno, feeder, tag, start, end, port, url, api)
    df = pre.get_data()
    if save == True:
        if place == "":
            df.to_csv('save/{_metric}.csv'.format(_metric=metric.replace(".","")))
        else:
            df.to_csv(f'save/{place}{typeno}{feeder}.csv')
    print("Data info")
    print("="*20)
    print(df.describe())
    print("="*20)
    print("Number of missing data before process :\n",df.isna().sum())
    print("="*20)
    return df

class Preprocessing:
    def __init__(self, metric, place, typeno, feeder, tag, start, end, port, url, api):
        self.metric = metric
        self.place = place
        self.typeno = typeno
        self.feeder = feeder
        self.tag = tag
        self.start = start
        self.end = end
        self.port = port
        self.api = api
        self.url = url
        if self.place == "":
            self.tags = f"tag:{self.tag}"
        else:
            self.tags = f"place:{self.place},typeno:{self.typeno},feeder:{self.feeder},tag:{self.tag}"
        
        self.dps = get_data(
            f"{self.url}"
            f":{self.port}" #tsdb 쿼리 프로세스 
            f"{self.api}" #api
            f"/{self.metric}" #메트릭 이름 설정
            f"?start={self.start}" #시작포인트
            f"&end={self.end}" #엔드포인트
            "&agg=sum" 
            f"&tags={self.tags}" # 태그 설정
            # '&downsample=5m-avg'  : 다운 샘플링
            # "&output=list" : 리스트로 값 리턴
            )

    def get_data(self):
        values = [d['dps'] for d in self.dps['results']]
        col_name =[d['tags']['tag'] for d in self.dps['results']]
        df = pd.DataFrame(values).T
        df = self.get_prc_data(df)
        df = self.re_index(df)
        df.columns = col_name
        return df
    
    def get_prc_data(self, data):
        # 입력 데이터가 비어 있는지 확인
        if data.empty:
            raise ValueError("Data is empty")

        data.index = data.index.astype(int)
        num_li = []
        
        for i in range(int(data.index[0]), int(data.index[-1])+5, 5):
            num_li.append(i)
        
        df = pd.DataFrame(num_li).set_index(0)
        result = df.join(data)

        return result


    def re_index(self, data):
        data.index = pd.to_datetime(data.index, unit='s', utc=True)
        data.index = data.index.tz_convert('Asia/Seoul').tz_localize(None)
        data.index.name = None
        # data.rename(columns={1:'data'}, inplace=True)
        return data
        

    def decode_data(self, data):
        data = data.dropna()
        data.index = data.index.astype(int) // 10**9
        data.index = data.index.astype(str)
        # data.columns = None
        df = data.astype(int)
        dict_from_df = df.to_dict()
        lst = list(dict_from_df.values())
        return lst

    def push_data(self, data):
        dps = self.dps
        for d, new_dps in zip(dps['results'], self.decode_data(data)):
            d['dps'] = new_dps
        #['dps'] = self.decode_data(data)

        return dps
    
