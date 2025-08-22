import os
import glob
import json
import datetime
import argparse
import math
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from krypton_utils.devmind import col2id

def read_feature_from_plan(plan_file):
    with open(plan_file, 'r', encoding='utf-8') as file:
        content = file.read()
        if "PLAN STEP FEATURE" not in content:
            return None
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
        plan_feature_json["query_id"] = plan_file.split('/')[-1].split('.')[0]
    except json.JSONDecodeError as e:
        # print(f"Error decoding JSON from {plan_file}: {e}")
        return None

    return plan_feature_json

def column_name_to_id(column_name: str):
    if column_name == "count()":
        return "count"
    if column_name == "literal":
        return "literal"
    if column_name == "rank()":
        return "rank"
    if "InputRefExpr" in column_name:
        return ""
    for col in col2id:
        if col.endswith(column_name.lower()):
            return col2id[col]
    for entry in column_name.split(' '):
        if entry.count('.') >= 2:
            col_name = entry.split('.')[-1]
            tab_name = entry.split('.')[-2]
            if f'{tab_name}.{col_name}' in col2id:
                return col2id[f'{tab_name}.{col_name}']
    
    # print(f"Column name {column_name} not found in col2id mapping.")
    return ""

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
        if r_literal.count(':') == 2:
            v = (datetime.datetime.strptime(r_literal, "%Y-%m-%d %H:%M:%S") - datetime.datetime(2020, 1, 1)).total_seconds()
        else:
            v = (datetime.datetime.strptime(r_literal, "%Y-%m-%d") - datetime.datetime(2020, 1, 1)).total_seconds()
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
    if "opName" not in plan_feature:
        return None
    if "analytic" in plan_feature["opName"].lower():
        normalized_plan_feature = normalize_plan_features(plan_feature["children"][0], literal_min_max)
        if 'filter_columns' in normalized_plan_feature['plan_parameters'] and "filterColumns" in plan_feature:
            normalized_plan_feature['plan_parameters']['filter_columns'] = {
                "column": None,
                "operator": "AND",
                "r_literal": {
                    "type": None,
                    "literal": None
                },
                "children": [
                    normalize_filter(plan_feature['filterColumns'], literal_min_max),
                    normalized_plan_feature['plan_parameters']['filter_columns']
                ]
            }
        elif "filterColumns" in plan_feature:
            normalized_plan_feature['plan_parameters']['filter_columns'] = normalize_filter(plan_feature['filterColumns'], literal_min_max)
        return normalized_plan_feature

    normalized_plan_feature = {
        "plan_parameters": {
            "op_name": plan_feature['opName'] if "join" not in plan_feature['opName'].lower() else "JoinStep",
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

    if 'nLimit' in plan_feature:
        normalized_plan_feature["plan_parameters"]['n_limit'] = plan_feature['nLimit']

    if 'setOpType' in plan_feature:
        normalized_plan_feature["plan_parameters"]['set_op_type'] = plan_feature['setOpType'].split("Step")[0].lower()

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
            if normalized_child is not None:
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
        if child is not None:
            normalize_literal(child, literal_min_max)

def split_subplans(plan_feature: dict):
    subplans = []
    stack = [plan_feature]
    while stack:
        if len(stack) == 0:
            break
        current_plan = stack.pop()
        current_plan["query_id"] = plan_feature["query_id"]
        if "read" not in current_plan["plan_parameters"]["op_name"].lower() and current_plan["plan_runtime"] != 0:
            subplans.append(current_plan)
        for child in current_plan.get('children', []):
            stack.append(child)
    return subplans

def read_plan_files(args):
    train_files = glob.glob(f"{args.input_dir}/devmind_train_plans/*.out")
    test_files = glob.glob(f"{args.input_dir}/devmind_test_plans/*.out")
    
    return train_files, test_files

def prepare_training_data(args):
    train_files, test_files = read_plan_files(args)

    train_plans = []
    test_plans = []
    literal_min_max = {}

    print(f"Processing training plan ...")
    for train_file in tqdm(train_files):
        plan_feature = read_feature_from_plan(train_file)
        if plan_feature is None:
            continue
        normalized_plan_feature = normalize_plan_features(plan_feature, literal_min_max)
    for train_file in tqdm(train_files):
        plan_feature = read_feature_from_plan(train_file)
        if plan_feature is None:
            continue
        normalized_plan_feature = normalize_plan_features(plan_feature, literal_min_max)
        if normalized_plan_feature is None:
            continue
        normalized_plan_feature["query_id"] = plan_feature["query_id"]
        normalize_literal(normalized_plan_feature, literal_min_max)
        subplans = split_subplans(normalized_plan_feature)
        train_plans.extend(subplans)

    print(f"Number of training plans: {len(train_plans)}")

    for test_file in tqdm(test_files):
        plan_feature = read_feature_from_plan(test_file)
        if plan_feature is None:
            continue
        normalized_plan_feature = normalize_plan_features(plan_feature, literal_min_max)
        if normalized_plan_feature is None:
            continue
        normalized_plan_feature["query_id"] = plan_feature["query_id"]
        normalize_literal(normalized_plan_feature, literal_min_max)
        subplans = split_subplans(normalized_plan_feature)
        test_plans.extend(subplans)

    print(f"Number of test plans: {len(test_plans)}")

    with open(f"krypton_utils/devmind_stats.json") as f:
        devmind_stats = json.load(f)

    test_data = {
        "parsed_plans": test_plans,
        "literal_min_max": literal_min_max,
        "database_stats": devmind_stats,
        "run_kwargs": {
            "hardware": "cpu"
        }
    }

    train_data = {
        "parsed_plans": train_plans,
        "literal_min_max": literal_min_max,
        "database_stats": devmind_stats,
        "run_kwargs": {
            "hardware": "cpu"
        }
    }

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    with open(os.path.join(args.output_dir, "devmind_train_data.json"), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.output_dir, "devmind_test_data.json"), 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.output_dir, "literal_min_max.json"), 'w', encoding='utf-8') as f:
        json.dump(literal_min_max, f, indent=2, ensure_ascii=False)

def prepare_eval_data(args):
    # load previous results
    existing_hashcodes = set()
    # with open(f"{args.output_dir}/model_inference.txt", 'r', encoding='utf-8') as f:
    #     for line in f:
    #         hashcode = line.strip().split(',')[0]
    #         existing_hashcodes.add(hashcode)

    with open(f"{args.output_dir}/literal_min_max.json", 'r', encoding='utf-8') as f:
        literal_min_max = json.load(f)
    
    eval_plan_features = []
    eval_plan_features_raw = []
    with open(f"{args.output_dir}/unique_plan_features.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            plan_feature = json.loads(line.strip())
            eval_plan_features_raw.append(plan_feature)
            normalized_plan_feature = normalize_plan_features(plan_feature, literal_min_max)
            normalize_literal(normalized_plan_feature, literal_min_max)
            eval_plan_features.append(normalized_plan_feature)

    print(f"Number of new eval plans : {len(eval_plan_features)}")

    with open(f"krypton_utils/devmind_stats.json") as f:
        devmind_stats = json.load(f)

    eval_data = {
        "parsed_plans": eval_plan_features,
        "literal_min_max": literal_min_max,
        "database_stats": devmind_stats,
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize plan features and filters from JSON files.")
    parser.add_argument('--input_dir', type=str, default="/mydata/workloads/devmind/", help='Input directory containing the plan feature files.')
    parser.add_argument('--output_dir', type=str, default="./data/devmind", help='Output file to save the normalized plan features.')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'finalize', 'confidence'], help='Mode to run the script: train or eval.')
    args = parser.parse_args()

    if args.mode == 'train':
        prepare_training_data(args)
    elif args.mode == 'eval':
        prepare_eval_data(args)
    elif args.mode == 'finalize':
        print("Finalizing data cleaning...")
        merge_prediction_hash(args)
    elif args.mode == 'confidence':
        print("Considering confidence...")
        consider_confidence()
