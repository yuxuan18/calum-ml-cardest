import os
import glob
import json
import datetime
import argparse
import math
from copy import deepcopy

from krypton_utils.tpch import col2id

def read_feature_from_plan(plan_file):
    with open(plan_file, 'r', encoding='utf-8') as file:
        content = file.read()
        plan_feature_lines = content.split('PLAN STEP FEATURE')[-1].strip().split("\n")
    plan_feature = ""
    for line in plan_feature_lines:
        if plan_feature.endswith('\t \t '):
            plan_feature = plan_feature[:-4]
        plan_feature += line.strip()
    # replace tab with \t
    plan_feature = plan_feature.replace('\t', '\\t')
    try:
        plan_feature_json = json.loads(plan_feature)
        plan_feature_json["query_id"] = plan_file.split('/')[-2].split('.')[0].split('query')[-1]
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {plan_file}: {e}")

    return plan_feature_json

def column_name_to_id(column_name: str):
    if column_name == "count()":
        return "count"
    for col in col2id:
        if col.endswith(column_name.lower()):
            return col2id[col]
    raise ValueError(f"Column name {column_name} not found in col2id mapping.")

def list2str(lst):
    if len(lst) == 0:
        return ""
    return ','.join(sorted(list(set([str(e) for e in lst]))))

def value2float(r_literal: str, r_type: str):
    if r_type == "integer":
        v = int(r_literal)
    elif r_type == "float":
        v = float(r_literal)
    elif r_type == "string":
        v = 1
    elif r_type == "string_like":
        v = r_literal.count('%')
    elif r_type == "set":
        v = int(r_literal)
    elif r_type == "datetime":
        v = (datetime.datetime.strptime(r_literal, "%Y-%m-%d") - datetime.datetime(1970, 1, 1)).total_seconds()
    elif r_type == "column":
        columns = [col.strip() for col in r_literal.split(',')]
        columns = [column_name_to_id(col) for col in columns]
        v = list2str(columns)
    else:
        raise ValueError(f"Unknown r_literal type: {r_type}")
    return v

def normalize_filter(filter_feature: dict, literal_min_max):
    if filter_feature['operator'].lower() in ['and', 'or'] and len(filter_feature['children']) == 1:
        return normalize_filter(filter_feature['children'][0], literal_min_max)

    normalized_filter_feature = {
        "operator": filter_feature['operator'],
        "column": None,
        "r_literal": None,
        "children": []
    }

    if 'column' in filter_feature:
        columns = [col.strip() for col in filter_feature['column'].split(',')]
        columns = [column_name_to_id(col) for col in columns]
        normalized_filter_feature["column"] = list2str(columns)

    if "rLiteralType" in filter_feature:
        normalized_filter_feature["r_literal"] = {
            "type": filter_feature['rLiteralType'],
            "literal": value2float(filter_feature['rLiteralValue'], filter_feature['rLiteralType'])
        }
    else:
        normalized_filter_feature["r_literal"] = {
            "type": None,
            "literal": None
        }

    if normalized_filter_feature["r_literal"]["type"] and normalized_filter_feature["r_literal"]["type"] != "column":
        if normalized_filter_feature["column"] not in literal_min_max:
            literal_min_max[normalized_filter_feature["column"]] = {
                "min": normalized_filter_feature["r_literal"]["literal"],
                "max": normalized_filter_feature["r_literal"]["literal"]
            }
        else:
            literal_min_max[normalized_filter_feature["column"]]["min"] = min(
                literal_min_max[normalized_filter_feature["column"]]["min"],
                normalized_filter_feature["r_literal"]["literal"]
            )
            literal_min_max[normalized_filter_feature["column"]]["max"] = max(
                literal_min_max[normalized_filter_feature["column"]]["max"],
                normalized_filter_feature["r_literal"]["literal"]
            )
    for child in filter_feature.get('children', []):
        normalized_child = normalize_filter(child, literal_min_max)
        normalized_filter_feature["children"].append(normalized_child)
    return normalized_filter_feature

def normalize_plan_features(plan_feature: dict, literal_min_max: dict):
    normalized_plan_feature = {
        "plan_parameters": {
            "op_name": plan_feature['opName'] if "join" not in plan_feature['opName'].lower() else "JoinStep",
            "est_card": plan_feature['estCard'] if 'estCard' in plan_feature else None,
        },
        "children": [],
        "plan_runtime": int(plan_feature["actCard"]) if "actCard" in plan_feature else 1,
    }

    # parse jointype
    if 'joinType' in plan_feature:
        normalized_plan_feature["plan_parameters"]['join_type'] = plan_feature['joinType']
    
    # parse est_card
    if 'estCard' in plan_feature:
        normalized_plan_feature["plan_parameters"]['est_card'] = plan_feature['estCard']

    # parse group_keys
    if 'groupKeys' in plan_feature:
        normalized_plan_feature["plan_parameters"]['group_keys'] = []
        for group_key_str in plan_feature['groupKeys']:
            group_key_id = column_name_to_id(group_key_str)
            normalized_plan_feature["plan_parameters"]['group_keys'].append(group_key_id)
        normalized_plan_feature["plan_parameters"]["group_keys"] = list2str(normalized_plan_feature["plan_parameters"]["group_keys"])
    
    # parse filter_columns
    if 'filterColumns' in plan_feature:
        normalized_plan_feature["plan_parameters"]['filter_columns'] = normalize_filter(plan_feature['filterColumns'], literal_min_max)
    
    # parse children
    if 'children' in plan_feature:
        for child in plan_feature['children']:
            normalized_child = normalize_plan_features(child, literal_min_max)
            normalized_plan_feature["children"].append(normalized_child)
    
    return normalized_plan_feature

def normalize_filter_literal(normalized_filter_feature: dict, literal_min_max: dict):
    if normalized_filter_feature["r_literal"] is not None:
        r_type = normalized_filter_feature["r_literal"]["type"]
        r_value = normalized_filter_feature["r_literal"]["literal"]
        if r_type != "column" and r_type:
            min_val = literal_min_max[normalized_filter_feature["column"]]["min"]
            max_val = literal_min_max[normalized_filter_feature["column"]]["max"]
            if r_value > literal_min_max[normalized_filter_feature["column"]]["max"]:
                print(f"r_literal value {r_value} out of bounds from [{min_val}, {max_val}] for column {normalized_filter_feature['column']}")
                r_value = 1
            elif r_value < literal_min_max[normalized_filter_feature["column"]]["min"]:
                print(f"r_literal value {r_value} out of bounds from [{min_val}, {max_val}] for column {normalized_filter_feature['column']}")
                r_value = 0
            elif literal_min_max[normalized_filter_feature["column"]]["max"] == literal_min_max[normalized_filter_feature["column"]]["min"]:
                r_value = 1
            else:
                r_value = (r_value - literal_min_max[normalized_filter_feature["column"]]["min"]) / \
                          (literal_min_max[normalized_filter_feature["column"]]["max"] - literal_min_max[normalized_filter_feature["column"]]["min"])
        normalized_filter_feature["r_literal"]["literal"] = r_value
    
    for child in normalized_filter_feature["children"]:
        normalize_filter_literal(child, literal_min_max)


def normalize_literal(normalized_plan_feature: dict, literal_min_max: dict):
    if "filter_columns" in normalized_plan_feature["plan_parameters"]:
        normalize_filter_literal(normalized_plan_feature["plan_parameters"]["filter_columns"], literal_min_max)
    
    for child in normalized_plan_feature["children"]:
        normalize_literal(child, literal_min_max)

def split_subplans(plan_feature: dict):
    subplans = []
    stack = [plan_feature]
    while stack:
        if len(stack) == 0:
            break
        current_plan = stack.pop()
        current_plan["query_id"] = plan_feature["query_id"]
        if "read" not in current_plan["plan_parameters"]["op_name"].lower():
            subplans.append(current_plan)
        for child in current_plan.get('children', []):
            stack.append(child)
    return subplans

def read_plan_files():
    test = [2, 3, 4, 5]
    train = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
    test_files = []
    train_files = []
    for i in train:
        train_files.extend(glob.glob(f"/mydata/workloads/tpch_1t_plans/query*/{i}.out"))
    for i in test:
        test_files.extend(glob.glob(f"/mydata/workloads/tpch_1t_plans/query*/{i}.out"))
    
    return train_files, test_files

def prepare_training_data(args):
    train_files, test_files = read_plan_files()

    train_plans = []
    test_plans = []
    literal_min_max = {}

    for train_file in train_files:
        plan_feature = read_feature_from_plan(train_file)
        normalized_plan_feature = normalize_plan_features(plan_feature, literal_min_max)
        normalized_plan_feature["query_id"] = plan_feature["query_id"]
        normalize_literal(normalized_plan_feature, literal_min_max)
        subplans = split_subplans(normalized_plan_feature)
        train_plans.extend(subplans)

    for test_file in test_files:
        plan_feature = read_feature_from_plan(test_file)
        normalized_plan_feature = normalize_plan_features(plan_feature, {})
        normalized_plan_feature["query_id"] = plan_feature["query_id"]
        normalize_literal(normalized_plan_feature, literal_min_max)
        subplans = split_subplans(normalized_plan_feature)
        test_plans.extend(subplans)

    with open(f"krypton_utils/tpch_stats.json") as f:
        tpch_stats = json.load(f)

    test_data = {
        "parsed_plans": test_plans,
        "literal_min_max": literal_min_max,
        "database_stats": tpch_stats,
        "run_kwargs": {
            "hardware": "cpu"
        }
    }

    train_data = {
        "parsed_plans": train_plans,
        "literal_min_max": literal_min_max,
        "database_stats": tpch_stats,
        "run_kwargs": {
            "hardware": "cpu"
        }
    }

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # with open(os.path.join(args.output_dir, "train_data.json"), 'w', encoding='utf-8') as f:
    #     json.dump(train_data, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.output_dir, "test_data.json"), 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    # with open(os.path.join(args.output_dir, "literal_min_max.json"), 'w', encoding='utf-8') as f:
    #     json.dump(literal_min_max, f, indent=2, ensure_ascii=False)

def prepare_eval_data(args):

    with open(f"{args.output_dir}/literal_min_max.json", 'r', encoding='utf-8') as f:
        literal_min_max = json.load(f)
    
    eval_plan_features = []
    eval_plan_features_raw = []
    with open(f"{args.output_dir}/unique_plan_features.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            plan_feature = json.loads(line.strip())
            eval_plan_features_raw.append(plan_feature)
            normalized_plan_feature = normalize_plan_features(plan_feature, {})
            normalize_literal(normalized_plan_feature, literal_min_max)
            eval_plan_features.append(normalized_plan_feature)

    print(f"Number of new eval plans : {len(eval_plan_features)}")

    with open(f"krypton_utils/tpch_stats.json") as f:
        tpch_stats = json.load(f)

    eval_data = {
        "parsed_plans": eval_plan_features,
        "literal_min_max": literal_min_max,
        "database_stats": tpch_stats,
        "run_kwargs": {
            "hardware": "cpu"
        }
    }

    with open(os.path.join(args.output_dir, "eval_data.json"), 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)


def merge_prediction_hash(args):
    hashcodes = []
    with open(f"{args.output_dir}/unique_plan_feature_hashcode.txt") as f:
        for line in f:
            hashcodes.append(line.strip())
    predictions = []
    with open(f"results.csv") as f:
        for line in f:
            predictions.append(math.ceil(float(line.strip().split(',')[1])))
    if len(hashcodes) != len(predictions):
        raise ValueError(f"Number of hashcodes {len(hashcodes)} does not match number of predictions {len(predictions)}")
    with open(f"{args.output_dir}/model_inference.txt", 'a+', encoding='utf-8') as f:
        for i in range(len(hashcodes)):
            f.write(f"{hashcodes[i]},{predictions[i]}\n")

def clean_plan(plan_feature):
    stack = [plan_feature]
    while stack:
        current_plan = stack.pop()
        if "children" in current_plan:
            for child in current_plan["children"]:
                stack.append(child)
        if "actCard" in current_plan:
            del current_plan["actCard"]
        if "join" in current_plan["opName"].lower():
            current_plan["opName"] = "JoinStep"

def collect_act_card(plan_feature):
    stack = [plan_feature]
    actCards = []
    while stack:
        current_plan = stack.pop()
        actCards.append(current_plan.get("actCard", None))
        if "children" in current_plan:
            for child in current_plan["children"]:
                stack.append(child)
    return actCards

def clean_and_split_subplans(plan_feature: dict):
    subplans = []
    actCards = collect_act_card(plan_feature)
    stack = [plan_feature]
    while stack:
        if len(stack) == 0:
            break
        current_plan = deepcopy(stack.pop())
        print(f"Processing plan: {current_plan}")
        clean_plan(current_plan)
        assert 'actCard' not in current_plan, "actCard should be removed from the plan feature"
        subplans.append(current_plan)
        for child in current_plan.get('children', []):
            stack.append(child)
    return subplans, actCards

def prepare_filter_columns_to_hash(filter_columns: dict):
    if "rLiteralType" in filter_columns and filter_columns["rLiteralType"] == "column":
        all_columns = filter_columns["column"].split(',') + filter_columns["rLiteralValue"].split(',')
        filter_columns["column"] = list2str(all_columns)
        del filter_columns["rLiteralValue"]
    
    for child in filter_columns.get('children', []):
        prepare_filter_columns_to_hash(child)

def hash_plan_feature(plan_feature):
    plan_feature_copy = deepcopy(plan_feature)
    # prepare the filter columns for hashing
    stack = [plan_feature_copy]
    while stack:
        if len(stack) == 0:
            break
        current_plan = stack.pop()
        if "plan_parameters" in current_plan and "filter_columns" in current_plan["plan_parameters"]:
            prepare_filter_columns_to_hash(current_plan["plan_parameters"]["filter_columns"])
        for child in current_plan.get('children', []):
            stack.append(child)
    return hash(json.dumps(plan_feature_copy, sort_keys=True, ensure_ascii=False))

def collect_feedback(args):
    # read unique plan features
    with open(f"{args.output_dir}/unique_plan_features.jsonl", 'r', encoding='utf-8') as f:
        unique_plan_features = [json.loads(line.strip()) for line in f]
    
    hashcodes = []
    for plan_feature in unique_plan_features:
        clean_plan(plan_feature)
        hashcode = hash_plan_feature(plan_feature)
        hashcodes.append(hashcode)
    
    # read model inference results
    est_cards = []
    with open(f"{args.output_dir}/model_inference.txt", 'r', encoding='utf-8') as f:
        for line in f:
            hashcode, est_card = line.strip().split(',')
            est_cards.append(float(est_card))

    # read runtime profile
    plan_files = glob.glob("/mydata/workloads/tpch_1t_test/model/query*/1.out")
    runtime_profiles = []
    for plan_file in plan_files:
        runtime_profiles.append(read_feature_from_plan(plan_file))

    # get all the subplans and their actCard
    subplans = []
    act_cards = []
    for plan_feature in runtime_profiles:
        plans, actCards = clean_and_split_subplans(plan_feature)
        subplans.extend(plans)
        act_cards.extend(actCards)
    if len(subplans) != len(act_cards):
        raise ValueError(f"Number of subplans {len(subplans)} does not match number of actCards {len(act_cards)}")
    
    subplan_hashcodes = []
    for plan_feature in subplans:
        subplan_hashcodes.append(hash_plan_feature(plan_feature))


    # read literal min max
    with open(f"{args.output_dir}/literal_min_max.json", 'r', encoding='utf-8') as f:
        literal_min_max = json.load(f)
    
    # search for the hashcodes in the unique plan features
    fail_to_find = 0
    finetune_plan_features = []
    finetune_plan_features_test = []
    for i, hashcde in enumerate(hashcodes):
        plan_feature = unique_plan_features[i]
        if hashcde in subplan_hashcodes:
            index = subplan_hashcodes.index(hashcde)
            act_card = float(act_cards[index]) if act_cards[index] is not None else -1
            est_card = est_cards[i]
            normalized_plan_feature = normalize_plan_features(plan_feature, {})
            normalize_literal(normalized_plan_feature, literal_min_max)
            normalized_plan_feature["plan_runtime"] = act_card
            if max(est_card / act_card, act_card / est_card) > 2:
                finetune_plan_features.append(normalized_plan_feature)
            else:
                finetune_plan_features_test.append(normalized_plan_feature)
        else:
            fail_to_find += 1

    print(f"Failed to find {fail_to_find} out of {len(hashcodes)} hashcodes in the runtime profile subplans.")

    with open(f"krypton_utils/tpch_stats.json") as f:
        tpch_stats = json.load(f)

    finetune_data = {
        "parsed_plans": finetune_plan_features,
        "literal_min_max": literal_min_max,
        "database_stats": tpch_stats,
        "run_kwargs": {
            "hardware": "cpu"
        }
    }

    with open(os.path.join(args.output_dir, "finetune_data.json"), 'w', encoding='utf-8') as f:
        json.dump(finetune_data, f, indent=2, ensure_ascii=False)

    finetune_data_test = {
        "parsed_plans": finetune_plan_features_test,
        "literal_min_max": literal_min_max,
        "database_stats": tpch_stats,
        "run_kwargs": {
            "hardware": "cpu"
        }
    }

    with open(os.path.join(args.output_dir, "finetune_data_test.json"), 'w', encoding='utf-8') as f:
        json.dump(finetune_data_test, f, indent=2, ensure_ascii=False)

def consider_confidence(args):
    is_below_threshold = []
    with open(f"{args.output_dir}/confidence.csv", 'r', encoding='utf-8') as f:
        for line in f:
            proxy_q_error, is_below = line.strip().split(',')
            is_below_threshold.append(is_below.lower() == 'true')
    
    hashcodes = []
    with open(f"{args.output_dir}/unique_plan_feature_hashcode.txt") as f:
        for line in f:
            hashcodes.append(line.strip())
    predictions = []
    with open(f"{args.output_dir}/eval_predictions.csv") as f:
        for line in f:
            predictions.append(math.ceil(float(line.strip().split(',')[1])))

    model_inference_w_confidence = []
    for is_below, hashcode, prediction in zip(is_below_threshold, hashcodes, predictions):
        if is_below:
            model_inference_w_confidence.append(f"{hashcode},{prediction}\n")

    with open(f"{args.output_dir}/model_inference_w_confidence.txt", 'w', encoding='utf-8') as f:
        f.writelines(model_inference_w_confidence)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize plan features and filters from JSON files.")
    parser.add_argument('--output_dir', type=str, default="./data/tpch", help='Output file to save the normalized plan features.')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'finalize', 'feedback', 'confidence'], help='Mode to run the script: train or eval.')
    args = parser.parse_args()

    if args.mode == 'train':
        prepare_training_data(args)
    elif args.mode == 'eval':
        prepare_eval_data(args)
    elif args.mode == 'finalize':
        print("Finalizing data cleaning...")
        merge_prediction_hash(args)
    elif args.mode == 'feedback':
        print("Collecting feedback...")
        collect_feedback(args)
    elif args.mode == 'confidence':
        print("Considering confidence...")
        consider_confidence(args)