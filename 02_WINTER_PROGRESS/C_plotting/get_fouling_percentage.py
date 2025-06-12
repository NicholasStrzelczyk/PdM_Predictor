import os
import numpy as np
import cv2
import pandas as pd

# Global Constants
SIG_DIGS = 9


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
    

if __name__ == "__main__":
    # Hyperparameters
    path_to_CGS_ds = '/Users/nick_1/PycharmProjects/UWO Masters/data_60/mar28_ds4' # CGS
    path_to_SMS_ds = '/Users/nick_1/PycharmProjects/UWO Masters/data_60/mar28_ds5' # SMS
    save_folder = '/Users/nick_1/PycharmProjects/UWO Masters/PdM_Predictor/03_ICML_Results'

    # calculate CGS fouling percentage
    calculate_CGS_fouling_percentage(path_to_CGS_ds, save_folder)

    # calculate SMS fouling percentage
    calculate_SMS_fouling_percentage(path_to_SMS_ds, save_folder)