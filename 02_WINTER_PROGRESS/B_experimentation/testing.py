import argparse
import os
from datetime import datetime

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import f1_score, precision, recall, jaccard_index, confusion_matrix
from tqdm import tqdm

from utils import *
from image_processing import preprocess_target, postprocess_seg_mask, visualize_seg_mask
from custom_ds_60 import Custom_DS_60
from custom_ds_60_SMS import Custom_DS_60_SMS

# ---------- Testing Methods ---------- #

def test_multiclass(model, test_loader, save_path, n_classes, scenario):
    metrics_history = {
        "Hour": [],
        "Day": [],
        "Class 0 F1 Score": [], "Class 1 F1 Score": [], "Class 2 F1 Score": [], "Class 3 F1 Score": [], "Class 4 F1 Score": [], "Class 5 F1 Score": [], "Class 6 F1 Score": [],
        "Class 0 Precision": [], "Class 1 Precision": [], "Class 2 Precision": [], "Class 3 Precision": [], "Class 4 Precision": [], "Class 5 Precision": [], "Class 6 Precision": [],
        "Class 0 Recall": [], "Class 1 Recall": [], "Class 2 Recall": [], "Class 3 Recall": [], "Class 4 Recall": [], "Class 5 Recall": [], "Class 6 Recall": [],
        "Class 0 Jaccard Index": [], "Class 1 Jaccard Index": [], "Class 2 Jaccard Index": [], "Class 3 Jaccard Index": [], "Class 4 Jaccard Index": [], "Class 5 Jaccard Index": [], "Class 6 Jaccard Index": [],
    }
    outputs_path = os.path.join(save_path, 'outputs', scenario)
    os.makedirs(outputs_path, exist_ok=True)

    model.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    # --- iterate through all test samples --- #
    start_time = datetime.now()
    log_and_print("{} starting testing...\n".format(datetime.now()))
    model.eval()
    with torch.no_grad():
        for day_num, hour_num, sample, target in tqdm(test_loader, desc="testing progress"):
            metrics_history["Day"].append(day_num.item())
            metrics_history["Hour"].append(hour_num[0])
            target = preprocess_target(target, model.use_rois)
            output = model(sample)
            target = postprocess_seg_mask(target, n_classes, model.use_rois)
            output = postprocess_seg_mask(output, n_classes, model.use_rois)

            # calculate metrics
            f1_scores = f1_score(output, target, task='multiclass', num_classes=n_classes, average="none").tolist()
            precision_scores = precision(output, target, task='multiclass', num_classes=n_classes, average="none").tolist()
            recall_scores = recall(output, target, task='multiclass', num_classes=n_classes, average="none").tolist()
            jac_scores = jaccard_index(output, target, task='multiclass', num_classes=n_classes, average="none").tolist()
            assert len(f1_scores) == len(precision_scores) == len(recall_scores) == len(jac_scores) == n_classes, "ERROR: f1_scores and jac_scores must have length n_classes"
            for class_idx in range(n_classes):
                metrics_history[f"Class {class_idx} F1 Score"].append(f1_scores[class_idx])
                metrics_history[f"Class {class_idx} Precision"].append(precision_scores[class_idx])
                metrics_history[f"Class {class_idx} Recall"].append(recall_scores[class_idx])
                metrics_history[f"Class {class_idx} Jaccard Index"].append(jac_scores[class_idx])

            # save segmentation masks
            seg_mask_path = os.path.join(outputs_path, "output_{}_{}.png".format(day_num.item(), hour_num[0]))
            cv2.imwrite(seg_mask_path, visualize_seg_mask(output, n_classes))
            del day_num, hour_num, sample, target, output

    # --- print results --- #
    log_and_print("\n{} mean testing metrics:".format(datetime.now()))
    for class_idx in range(n_classes):
        log_and_print("[class {}] f1_score: {:.9f} +/- {:.5f}, precision: {:.9f} +/- {:.5f}, recall: {:.9f} +/- {:.5f}, jaccard_idx: {:.9f} +/- {:.5f}".format(
            class_idx, 
            np.mean(metrics_history[f"Class {class_idx} F1 Score"]), np.std(metrics_history[f"Class {class_idx} F1 Score"]), 
            np.mean(metrics_history[f"Class {class_idx} Precision"]), np.std(metrics_history[f"Class {class_idx} Precision"]),
            np.mean(metrics_history[f"Class {class_idx} Recall"]), np.std(metrics_history[f"Class {class_idx} Recall"]),
            np.mean(metrics_history[f"Class {class_idx} Jaccard Index"]), np.std(metrics_history[f"Class {class_idx} Jaccard Index"])))

    # --- save metrics --- #
    total_testing_time = datetime.now() - start_time
    log_and_print("\n{} testing complete.".format(datetime.now()))
    log_and_print("total testing time: {}".format(total_testing_time))
    log_and_print("{} saving metrics and generating plots...".format(datetime.now()))
    df = pd.DataFrame(metrics_history)
    df.to_csv(os.path.join(save_path, f'{scenario}_testing_metrics.csv'), index=False)
    log_and_print("{} testing script finished.\n".format(datetime.now()))


def test_binary(model, test_loader, save_path, n_classes, scenario):
    metrics_history = {
        "Hour": [], 
        "Day": [],
        "F1 Score": [], 
        "Precision": [], 
        "Recall": [], 
        "Jaccard Index": [],
        "TN": [], 
        "FP": [], 
        "FN": [], 
        "TP": []
    }
    outputs_path = os.path.join(save_path, 'outputs', scenario)
    os.makedirs(outputs_path, exist_ok=True)

    model.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    # --- iterate through all test samples --- #
    start_time = datetime.now()
    log_and_print("{} starting testing...\n".format(datetime.now()))
    model.eval()
    with torch.no_grad():
        for day_num, hour_num, sample, target in tqdm(test_loader, desc="testing progress"):
            metrics_history["Day"].append(day_num.item())
            metrics_history["Hour"].append(hour_num[0])
            target = preprocess_target(target, model.use_rois)
            output = model(sample)
            target = postprocess_seg_mask(target, n_classes, model.use_rois)
            output = postprocess_seg_mask(output, n_classes, model.use_rois)

            # calculate metrics
            metrics_history["F1 Score"].append(f1_score(output, target, task='binary').item())
            metrics_history["Precision"].append(precision(output, target, task='binary').item())
            metrics_history["Recall"].append(recall(output, target, task='binary').item())
            metrics_history["Jaccard Index"].append(jaccard_index(output, target, task='binary').item())
            conf_mat = confusion_matrix(output, target, task='binary').tolist()
            metrics_history["TN"].append(conf_mat[0][0])
            metrics_history["FP"].append(conf_mat[0][1])
            metrics_history["FN"].append(conf_mat[1][0])
            metrics_history["TP"].append(conf_mat[1][1])

            # save segmentation masks
            seg_mask_path = os.path.join(outputs_path, "output_{}_{}.png".format(day_num.item(), hour_num[0]))
            cv2.imwrite(seg_mask_path, visualize_seg_mask(output, n_classes))
            del day_num, hour_num, sample, target, output

    # --- print results --- #
    log_and_print("\n{} mean testing metrics:".format(datetime.now()))
    log_and_print("f1_score: {:.9f} +/- {:.5f}, precision: {:.9f} +/- {:.5f}, recall: {:.9f} +/- {:.5f}, jaccard_idx: {:.9f} +/- {:.5f}".format(
        np.mean(metrics_history["F1 Score"]), np.std(metrics_history["F1 Score"]), 
        np.mean(metrics_history["Precision"]), np.std(metrics_history["Precision"]),
        np.mean(metrics_history["Recall"]), np.std(metrics_history["Recall"]),
        np.mean(metrics_history["Jaccard Index"]), np.std(metrics_history["Jaccard Index"])))

    # --- save metrics --- #
    total_testing_time = datetime.now() - start_time
    log_and_print("\n{} testing complete.".format(datetime.now()))
    log_and_print("total testing time: {}".format(total_testing_time))
    log_and_print("{} saving metrics and generating plots...".format(datetime.now()))
    df = pd.DataFrame(metrics_history)
    df.to_csv(os.path.join(save_path, f'{scenario}_testing_metrics.csv'), index=False)
    log_and_print("{} testing script finished.\n".format(datetime.now()))

# ---------- Main Method ---------- #

if __name__ == "__main__":
    # get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, required=True, help="model name (str)")
    parser.add_argument("-rois", type=str, required=True, help="use rois (y/n)")
    parser.add_argument("-binary", type=str, required=True, help="use binary targets (y/n)")
    parser.add_argument("-dataset", type=str, required=True, help="dataset folder name (str)")
    parser.add_argument("-scenario", type=str, required=True, help="dataset scenario type (str)")
    parser.add_argument("-trial", type=int, required=True, help="trial number (int)")
    args = parser.parse_args()

    # get hyperparameters
    model_name = args.model.lower()
    use_rois = args.rois.lower() == "y"
    binary_targets = args.binary.lower() == "y"
    dataset_name = args.dataset.lower()
    scenario = args.scenario.upper()
    assert scenario in ['SMS', 'CGS'], "ERROR: scenario must be either SMS or CGS"
    trial = args.trial
    classes = 2 if binary_targets else 7

    # set up save path
    results_folder_name = f"{model_name}_{'rois' if use_rois else 'full'}_{'binary' if binary_targets else 'multiclass'}"
    save_location = os.path.join(".", "RESULTS", f"trial_{trial}", results_folder_name)

     # set up logger
    setup_basic_logger(save_location, f'testing_{scenario}')
    log_and_print(f"\n--- Testing {results_folder_name} | {scenario} | trial {trial} ---\n")

    # set up data loaders
    if scenario == 'SMS': 
        test_ds = Custom_DS_60_SMS(dataset_name, binary_targets) # for SMS Testing
    else:
        test_ds = Custom_DS_60(dataset_name, 'test', binary_targets) # for CGS Testing
    test_ds_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # set up model and load weights
    model = get_model(model_name, num_classes=classes, use_rois=use_rois)
    model.load_state_dict(torch.load(os.path.join(save_location, "best_weights.pth"), weights_only=True))

    # test model
    if binary_targets:
        test_binary(model, test_ds_loader, save_location, classes, scenario)
    else:
        test_multiclass(model, test_ds_loader, save_location, classes, scenario)