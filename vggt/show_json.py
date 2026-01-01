import json
import gzip


# full dataset
file_path = '/home/suno/log1/_datasets/co3d_full/apple/set_lists/set_lists_manyview_dev_0.json'
# file_path = '/home/suno/log1/_datasets/co3d_full/apple/set_lists/set_lists_fewview_dev.json'

# sub dataset
# file_path = '/home/suno/log1/_datasets/co3d_sub/apple/set_lists/set_lists_manyview_dev_0.json'


with open(file_path, 'rt', encoding='utf-8') as f:
    data = json.load(f)

    if isinstance(data, list):
        print(f"*** Data type   : {type(data)}")
        print(f"*** Data len    : {len(data)}")

    elif isinstance(data, dict):
        print(f"*** Data type   : {type(data)}")
        print(f"*** Keys        : {data.keys()}")

        # key별 value 개수 및 value 형태
        for key, value in data.items():
            print(f"{key}   : {len(value)} {value[0]}")
