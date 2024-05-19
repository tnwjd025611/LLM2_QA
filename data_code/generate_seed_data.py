import random
import os
import json
from tqdm import tqdm

with open('./data/train_all.json','r') as f: # 이부분 수정할것
    data = json.load(f)

data_all = list()
count = 0

for i in tqdm(range(len(data))):
    for j in range(len(data[i])):
        data_all.append(data[i][f'{j}'])

SUBSAMPLE_SPLIT = 0.01

subsampled_data = random.sample(data_all, int(SUBSAMPLE_SPLIT * len(data)))

formatted_data = []
for dp in subsampled_data:
    # dp = eval(dp)
    formatted_data.append({
        "template_id": dp['template_id'],
        "question_type": dp["question_type"],
        "attribute_type": dp['attribute_type'],
        "ecg_id": dp['ecg_id'],
        "attribute": dp["attribute"],
        "description": dp['description'],
        "instruction" : dp['question'],
        "output" : dp['answer'],
        "input": "",
        "question_type" : "ecg",
        "use_cot" : True
    })

if not os.path.exists('data'):
    os.makedirs('data')

with open(f'data/seed/ecg_{SUBSAMPLE_SPLIT}.json','w') as f:
    json.dump(formatted_data, f)