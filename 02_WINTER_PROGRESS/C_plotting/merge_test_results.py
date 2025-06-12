from datetime import datetime
import pandas as pd
import os


# Global Constants
MODELS = ['cgnet', 'unet']
INPUT_TYPES = ['rois', 'full']
TARGET_TYPES = ['binary', 'multiclass']
SCENARIOS = ['SMS', 'CGS']


def merge_results(path_to_results, save_folder, target_type):
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
    

if __name__ == "__main__":
    # Hyperparameters
    path_to_results = '/Users/nick_1/PycharmProjects/UWO Masters/ICML_Jun12_Results'
    save_folder = '/Users/nick_1/PycharmProjects/UWO Masters/PdM_Predictor/03_ICML_Results'

    for target_type in TARGET_TYPES:
        merge_results(path_to_results, save_folder, target_type)

    print("Done - results merged")
    
    

