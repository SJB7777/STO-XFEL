import datetime
import numpy as np
runs = [11,12, 40, 41
,42
,43
,44
,45
,66
,67
,68
,69
,72
,73
,76
,77
,78
,82
,83
,86
,87
,88
,89
,94
,95
,96
,107
,108
,112
,113
,123
,124
,125
,126
,135
,136
,139
,140

,146
,147
,150
,151
,154
,155
,159
,160]

dts = []

for run in runs:

    log_file = f'/xfel/ffs/dat/ue_240821_FXS/log/type=measurement/run={run:03}/scan=001/p0001.h5.log'

    with open(log_file) as f:
        line = f.readline()
        times = line.split('  ')[0]
        a = times.split()
        timestr = '20' + a[1] + a[2]
        dt = datetime.datetime.strptime(timestr, '%Y%m%d%H:%M:%S')
        dts.append(dt)

dts = np.array(dts)
dts = dts - [dts[0]]
for dt in dts:
    print(dt.total_seconds())