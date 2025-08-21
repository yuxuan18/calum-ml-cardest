import json
import os

features = []
with open("plan_features.jsonl") as f:
    for line in f:
        line = line.replace('\\u', '\\\\u')
        features.append(line)

hash_code = []
with open("plan_feature_hashcode.txt") as f:
    for line in f:
        line = line.strip()
        hash_code.append(line)

unique_hash_code = []
unique_plan_features = []
for feature, hashcode in zip(features, hash_code):
    if hashcode not in unique_hash_code:
        unique_hash_code.append(hashcode)
        unique_plan_features.append(feature)

with open("unique_plan_features.jsonl", "w") as f:
    for feature in unique_plan_features:
        f.write(feature)

with open("unique_plan_feature_hashcode.txt", "w") as f:
    for hashcode in unique_hash_code:
        f.write(hashcode + "\n")