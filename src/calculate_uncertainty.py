import argparse
import numpy as np
import pandas as pd
import os

def read_data(args):
    train_predictions = pd.read_csv(args.train_prediction_file, header=None, names=["truecard", "prediction"])
    train_embeddings = np.load(args.train_embedding_file)
    test_embeddings = np.load(args.test_embedding_file)
    return train_predictions, train_embeddings, test_embeddings

def calculate_uncertainty(train_qerrors, train_embeddings, test_embeddings, k_ratio):
    # calculate the cosine similarity between training and test embeddings
    similarities = np.dot(test_embeddings, train_embeddings.T) / (
        np.linalg.norm(test_embeddings, axis=1, keepdims=True) * 
        np.linalg.norm(train_embeddings, axis=1, keepdims=True).T
    )
    # find the k nearest neighbors based on cosine similarity
    k = max(int(len(train_embeddings) * k_ratio), 1)
    print(f"Using k={k} for kNN calculation.")
    nearest_indices = np.argsort(-similarities, axis=1)[:, :k]
    nearest_qerrors = np.log(np.array(train_qerrors)[nearest_indices])
    nearest_similarities = similarities[np.arange(len(test_embeddings))[:, None], nearest_indices]
    # calculate the uncertainty as the weighted average of the nearest qerrors
    uncertainty = np.exp(np.sum(nearest_qerrors * nearest_similarities, axis=1) / np.sum(nearest_similarities, axis=1))
    return uncertainty

def calculate_bound(sample, confidence_level=0.95):
    mean = np.mean(sample)
    std_dev = np.std(sample, ddof=1)  # use sample standard deviation
    n = len(sample)
    delta = (1- confidence_level) / 2
    ub = mean + std_dev / np.sqrt(n) * np.sqrt(2 * np.log(1 / delta))
    lb = mean - std_dev / np.sqrt(n) * np.sqrt(2 * np.log(1 / delta))
    return lb, ub
    
def valid_recall(threshold, uncertainty, qerrors, max_qerror):
    # calculate recall
    predicted_positive = uncertainty > threshold
    true_positive = qerrors > max_qerror
    print(f"Predicted positive: {np.sum(predicted_positive)}, True positive: {np.sum(true_positive)}")
    recall = np.sum(predicted_positive & true_positive) / np.sum(true_positive)
    return recall


def calculate_uncertainty_threshold(train_qerrors, train_embeddings, validation_qerrors, validation_embeddings, args):
    # calculate the uncertainty for validation set
    validation_uncertainty = calculate_uncertainty(train_qerrors, train_embeddings, validation_embeddings, args.k_ratio)
    corr = np.corrcoef(np.log(validation_uncertainty), np.log(validation_qerrors))[0, 1]
    print(f"Correlation between validation uncertainty and Q errors: {corr}")
    
    # for a threshold of uncertainty
    # true positive is (uncertainty > threshold) and (qerror > max_qerror)
    # false positive is (uncertainty > threshold) and (qerror <= max_qerror)
    # false negative is (uncertainty <= threshold) and (qerror > max_qerror)
    # true negative is (uncertainty <= threshold) and (qerror <= max_qerror)
    # given a recall targeet, calculate an initial threshold
    sorted_indices = np.argsort(validation_uncertainty)[::-1]  # sort in descending order
    sorted_qerrors = np.array(validation_qerrors)[sorted_indices]
    sorted_uncertainty = validation_uncertainty[sorted_indices]
    total_positive = np.sum(sorted_qerrors > args.max_qerror)
    target_positive = int(total_positive * args.target_recall)
    print(f"Total positive samples: {total_positive}, Target positive samples: {target_positive}")
    threshold_index = np.where(np.cumsum(sorted_qerrors > args.max_qerror) >= target_positive)[0][0]
    initial_threshold = sorted_uncertainty[threshold_index]
    print(f"Initial uncertainty threshold: {initial_threshold}")
    print(f"Initial recall: {valid_recall(initial_threshold, validation_uncertainty, validation_qerrors, args.max_qerror)}")
    # refine the recall target using the confidence interval
    true_positives = (validation_uncertainty > initial_threshold) & (validation_qerrors > args.max_qerror)
    false_negatives = (validation_uncertainty <= initial_threshold) & (validation_qerrors > args.max_qerror)
    _, true_positive_rate_ub = calculate_bound(true_positives, args.probability)
    false_negative_rate_lb, _ = calculate_bound(false_negatives, args.probability)
    refined_recall = true_positive_rate_ub / (true_positive_rate_ub + false_negative_rate_lb)
    print(f"Refined recall: {refined_recall}")
    # recalculate the threshold based on the refined recall
    target_positive = int(total_positive * refined_recall)
    threshold_index = np.where(np.cumsum(sorted_qerrors > args.max_qerror) >= target_positive)[0][0]
    final_threshold = sorted_uncertainty[threshold_index]
    print(f"Final uncertainty threshold: {final_threshold}")
    print(f"Final recall: {valid_recall(final_threshold, validation_uncertainty, validation_qerrors, args.max_qerror)}")
    return final_threshold
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate uncertainty threshold.")
    parser.add_argument("--train_prediction_file", type=str, required=True, help="Path to the training prediction file.")
    parser.add_argument("--train_embedding_file", type=str, required=True, help="Path to the training embedding file.")
    parser.add_argument("--test_embedding_file", type=str, required=True, help="Path to the test embedding file.")
    parser.add_argument("--test_uncertainty_file", type=str, default=None, help="Path to save test uncertainty file.")
    parser.add_argument("--max_qerror", type=float, default=25, help="Maximum quantization error.")
    parser.add_argument("--target_recall", type=float, default=0.8, help="Target recall for uncertainty threshold calculation.")
    parser.add_argument("--probability", type=float, default=0.95, help="Probability for uncertainty threshold calculation.")
    parser.add_argument("--k_ratio", type=float, default=0.0005, help="Ratio of the number of data for kNN calculation.")
    parser.add_argument("--validation_ratio", type=int, default=0.2, help="Size of the validation set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--mode", type=str, choices=["uncertainty", "threshold"], default="threshold", help="Mode of operation: train or test.")
    args = parser.parse_args()

    if args.mode == "uncertainty":
        if os.path.exists(args.test_uncertainty_file):
            print(f"Test uncertainty file {args.test_uncertainty_file} already exists. Skipping calculation.")
            exit(0)

        train_predictions, train_embeddings, test_embeddings = read_data(args)
        train_qerrors = np.maximum(train_predictions["prediction"] / train_predictions["truecard"], train_predictions["truecard"] / train_predictions["prediction"])
        uncertainty = calculate_uncertainty(train_qerrors, train_embeddings, test_embeddings, args.k_ratio)
        with open(args.test_uncertainty_file, 'w') as f:
            for u in uncertainty:
                f.write(f"{u}\n")
    elif args.mode == "threshold":
        train_predictions, train_embeddings, test_embeddings = read_data(args)

        # set random seed for reproducibility
        np.random.seed(args.seed)

        # select validation data
        validation_indices = np.random.choice(
            range(len(train_predictions)),
            size=int(args.validation_ratio * len(train_predictions)),
            replace=False
        )
        validation_predictions = train_predictions.iloc[validation_indices]
        validation_embeddings = train_embeddings[validation_indices]
        train_predictions = train_predictions.drop(validation_indices)
        train_embeddings = np.delete(train_embeddings, validation_indices, axis=0)
        train_qerrors = np.maximum(train_predictions["prediction"] / train_predictions["truecard"], train_predictions["truecard"] / train_predictions["prediction"])
        validation_qerrors = np.maximum(validation_predictions["prediction"] / validation_predictions["truecard"], validation_predictions["truecard"] / validation_predictions["prediction"])

        threshold = calculate_uncertainty_threshold(
            train_qerrors, train_embeddings, validation_qerrors, validation_embeddings, args
        )

        print(f"Calculated uncertainty threshold: {threshold}")

        if args.test_uncertainty_file is None:
            print("No test uncertainty file provided, skipping test uncertainty calculation.")
            exit(0)

        if os.path.exists(args.test_uncertainty_file):
            test_uncertainty = []
            with open(args.test_uncertainty_file, 'r') as f:
                for line in f:
                    test_uncertainty.append(float(line.strip().split(',')[0]))
            test_uncertainty = np.array(test_uncertainty)
        else:
            test_uncertainty = calculate_uncertainty(
                train_qerrors, train_embeddings, test_embeddings, args.k_ratio
            )
        is_below_threshold = test_uncertainty < threshold
    
        with open(args.test_uncertainty_file, 'w') as f:
            for u, below in zip(test_uncertainty, is_below_threshold):
                f.write(f"{u},{below}\n")

