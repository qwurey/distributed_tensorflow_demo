## Distributed tensorflow application

#### Usage

```
python calc_w_b.py --ps_hosts=192.168.1.103:2222,192.168.1.103:2223 --worker_hosts=192.168.1.103:2224,192.168.1.103:2225 --job_name=ps --task_index=0

python calc_w_b.py --ps_hosts=192.168.1.103:2222,192.168.1.103:2223 --worker_hosts=192.168.1.103:2224,192.168.1.103:2225 --job_name=ps --task_index=1

python calc_w_b.py --ps_hosts=192.168.1.103:2222,192.168.1.103:2223 --worker_hosts=192.168.1.103:2224,192.168.1.103:2225 --job_name=worker --task_index=0

python calc_w_b.py --ps_hosts=192.168.1.103:2222,192.168.1.103:2223 --worker_hosts=192.168.1.103:2224,192.168.1.103:2225 --job_name=worker --task_index=1
```




