import os
from datetime import datetime
import numpy as np

base_path = os.getcwd()
base_path = os.path.join(base_path, 'JUNO')


datas = np.array([])
for filename in os.listdir((os.path.join(base_path, 'data/MUR_single_days'))):
    aux = filename.partition('_')[2]
    aux = aux.partition('.')[0]
    date = aux[:4] + '-' + aux[4:6] + '-' + aux[6:]
    datas = np.append(datas, date)

datas.sort()
dates_list = [datetime.strptime(date, "%Y-%m-%d").date() for date in datas]


for i in range(len(dates_list)-1):
    if int(str(dates_list[i+1] - dates_list[i])[0]) > 1:
        print(dates_list[i])

