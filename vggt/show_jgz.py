import json
import gzip


file_path = '/home/suno/log1/_datasets/co3d_annotation/apple_train.jgz'

with gzip.open(file_path, 'rt', encoding='utf-8') as f:
    data = json.load(f)

    if isinstance(data, list):
        print(f"*** Data type   : {type(data)}")
        print(f"*** Data len    : {len(data)}")

    elif isinstance(data, dict):
        print(f"*** Data type   : {type(data)}")
        print(f"*** Keys len    : {len(data)}")

        for key, value in data.items():
            print(f"*** Values len  : {len(value)}")
            print(f"Keys    : {key}")
            print(f"values  : {value[0]}")
            break
