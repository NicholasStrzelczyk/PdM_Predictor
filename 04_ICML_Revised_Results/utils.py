from datetime import datetime
import pandas as pd
import os
import cv2
import numpy as np

# Global Constants
SIG_DIGS = 9
MODELS = ['cgnet', 'unet']
INPUT_TYPES = ['rois', 'full']
TARGET_TYPES = ['binary', 'multiclass']
SCENARIOS = ['SMS', 'CGS']

# --------------------------------------------------------------- #
# ----------------------- Merging Results ----------------------- #
# --------------------------------------------------------------- #

def merge_results_for_target_type(path_to_results, save_folder, target_type):
    assert target_type in TARGET_TYPES, "ERROR: target_type must be either binary or multiclass"
    if target_type == 'binary':
        headers = ['Scenario', 'CNN Model', 'Input Type', 'Target Type', 'Trial', 'Day', 'Hour', 'F1 Score', 'Jaccard Index', 'TP', 'FP', 'FN', 'TN']
    else:
        headers = ['Scenario', 'CNN Model', 'Input Type', 'Target Type', 'Trial', 'Day', 'Hour']
        for metric in ['F1 Score', 'Jaccard Index']:
            for class_idx in range(7):
                headers.append(f'Class {class_idx} {metric}')
    
    df = pd.DataFrame(columns=headers)
    for trial in range(1, 4):
        for model in MODELS:
            for input_type in INPUT_TYPES:
                for scenario in SCENARIOS:
                    csv_path = os.path.join(path_to_results, f'trial_{trial}', f'{model}_{input_type}_{target_type}', f'{scenario}_testing_metrics.csv')
                    if not os.path.exists(csv_path):
                        print(f"ERROR: File not found: {csv_path}")
                        continue
                    current_df = pd.read_csv(csv_path)
                    current_df['Scenario'] = scenario
                    current_df['CNN Model'] = model
                    current_df['Input Type'] = input_type
                    current_df['Target Type'] = target_type
                    current_df['Trial'] = trial
                    # current_df.rename(columns={'Day': 'Hour', 'Hour': 'Day'}, inplace=True)
                    current_df = current_df[headers]
                    df = pd.concat([df, current_df], ignore_index=True)

    df.to_csv(os.path.join(save_folder, f'{target_type}_results.csv'), index=False)


def merge_results(path_to_results, save_folder):
    assert os.path.isdir(path_to_results), "ERROR: path_to_results must be a directory"
    assert os.path.isdir(save_folder), "ERROR: save_folder must be a directory"
    for target_type in TARGET_TYPES:
        merge_results_for_target_type(path_to_results, save_folder, target_type)
    print("Done - results merged")

# --------------------------------------------------------------- #
# ------------------ Training Time Calculation ------------------ #
# --------------------------------------------------------------- #

def calculate_training_time_trial1(log_file_path):
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()

        start_time_str = None
        end_time_str = None

        for line in lines:
            if 'lowest loss:' in line:
                end_epoch = int(line.split(' ')[-1])
                break

        for line in lines:
            if 'starting training...' in line:
                # Extract timestamp from the line like '2025-03-31 03:15:33.590674 starting training...'
                start_time_str = line.split(' ')[0] + ' ' + line.split(' ')[1]
            elif f'epoch {end_epoch}/25 metrics:' in line:
                # Extract timestamp from the line like '2025-03-31 03:41:16.651587 training complete.'
                end_time_str = line.split(' ')[0] + ' ' + line.split(' ')[1]
                break # Assuming 'training complete.' signifies the end and appears once

        if start_time_str and end_time_str:
            # Parse the timestamps
            start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S.%f')
            end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S.%f')

            # Calculate the difference
            diff = end_time - start_time
            return str(diff)
        else:
            return "Could not find start and/or end training timestamps in the log file."

    except FileNotFoundError:
        return f"Error: The file '{log_file_path}' was not found."
    except Exception as e:
        return f"An error occurred: {e}"
    

def calculate_training_time_other(log_file_path):
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if 'model training time:' in line:
                training_time_str = line.split(' ')[-1].strip()
                break

        return training_time_str

    except FileNotFoundError:
        print(f"Error: The file '{log_file_path}' was not found.")
        return f"Error: The file '{log_file_path}' was not found."  


def create_training_time_csv(path_to_results, save_folder):
    df = pd.DataFrame(columns=['model', 'roi_mode', 'target_mode', 'trial', 'training_time'])

    # get the training time for each model, roi mode, and target mode
    for trial in range(1, 4):
        for model in MODELS:
            for roi_mode in INPUT_TYPES:
                for target_mode in TARGET_TYPES:
                    log_file_path = os.path.join(path_to_results, f'trial_{trial}', f'{model}_{roi_mode}_{target_mode}', 'training.log')
                    if not os.path.exists(log_file_path):
                        training_time = 'N/A'
                    else:
                        if trial == 1:
                            training_time = calculate_training_time_trial1(log_file_path)
                        else:
                            training_time = calculate_training_time_other(log_file_path)

                    row_dict = {'model': model, 'roi_mode': roi_mode, 'target_mode': target_mode, 'trial': trial, 'training_time': training_time}
                    df = pd.concat([df, pd.DataFrame(row_dict, index=[0])], ignore_index=True)

    # save the dataframe to a CSV file
    df.to_csv(os.path.join(save_folder, 'training_time_plot.csv'), index=False)
    print("Done - training time csv created")

# --------------------------------------------------------------- #
# ------------------ Fouling Percentage Calculation ------------- #
# --------------------------------------------------------------- #

def calculate_CGS_fouling_percentage(path_to_ds, save_folder):
    targets_path = os.path.join(path_to_ds, 'test', 'targets', 'binary')
    save_file_name = 'CGS_fouling_percentage.csv'

    # create new csv file
    df = pd.DataFrame(columns=['day', 'background_percentage', 'fouling_percentage'])

    for day in range(1, 61):
        img = cv2.imread(os.path.join(targets_path, f'target_{day}.png'))
        # get percentage of pixels that are 0
        background_percentage = np.sum(img == 0) / img.size * 100
        # get percentage of pixels that are not 0
        fouling_percentage = np.sum(img != 0) / img.size * 100
        row_dict = {'day': day, 'background_percentage': np.round(background_percentage, SIG_DIGS), 'fouling_percentage': np.round(fouling_percentage, SIG_DIGS)}
        df = pd.concat([df, pd.DataFrame(row_dict, index=[0])], ignore_index=True)

    df.to_csv(os.path.join(save_folder, save_file_name), index=False)
    print("Done - CGS fouling percentage csv created")


def calculate_SMS_fouling_percentage(path_to_ds, save_folder):
    save_file_name = 'SMS_fouling_percentage.csv'

    # create new csv file
    df = pd.DataFrame(columns=['day', 'background_percentage', 'fouling_percentage'])

    for idx, partition in enumerate(['train', 'val', 'test']):
        targets_path = os.path.join(path_to_ds, partition, 'targets', 'binary')

        for day in range(1, 21):
            img = cv2.imread(os.path.join(targets_path, f'target_{day}.png'))
            # get percentage of pixels that are 0
            background_percentage = np.sum(img == 0) / img.size * 100
            # get percentage of pixels that are not 0
            fouling_percentage = np.sum(img != 0) / img.size * 100
            row_dict = {'day': day + (idx * 20), 'background_percentage': np.round(background_percentage, SIG_DIGS), 'fouling_percentage': np.round(fouling_percentage, SIG_DIGS)}
            df = pd.concat([df, pd.DataFrame(row_dict, index=[0])], ignore_index=True)

    df.to_csv(os.path.join(save_folder, save_file_name), index=False)
    print("Done - SMS fouling percentage csv created")

