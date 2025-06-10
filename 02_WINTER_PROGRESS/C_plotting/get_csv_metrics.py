import pandas as pd
import numpy as np

path_to_csv_1 = '/Users/nick_1/PycharmProjects/UWO Masters/PdM_Predictor/03_ICML_Results/CGS_binary_results.csv'
path_to_csv_2 = '/Users/nick_1/PycharmProjects/UWO Masters/PdM_Predictor/03_ICML_Results/SMS_binary_results.csv'

df_1 = pd.read_csv(path_to_csv_1)
# print(df_1.head())
mean_f1_score = df_1['F1 Score'].mean()
print(f"Average F1 Score CGS: {mean_f1_score:.4f}")

df_2 = pd.read_csv(path_to_csv_2)
# print(df_2.head())
mean_f1_score = df_2['F1 Score'].mean()
print(f"Average F1 Score SMS: {mean_f1_score:.4f}")