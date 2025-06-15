import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from sklearn.metrics import ConfusionMatrixDisplay
import gzip
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc, precision_recall_curve
from scipy.stats import pearsonr, spearmanr
import sys

def load_experiment_data(instance_path):
    dfs = {
        "pathways": None,
        "statistics": None,
        "train_stats": None,
        "test_stats": None, 
        "val_stats": None
    }

    if not os.path.isdir(instance_path):
        print(f"Error: Directory not found at {instance_path}")
        return tuple(dfs.values())

    for filename in os.listdir(instance_path):
        file_path = os.path.join(instance_path, filename)
        
        # Skip if not a file
        if not os.path.isfile(file_path):
            continue

        try:
            if 'analysis' in filename and filename.endswith('.json.gz'):
                print(f"Attempting to load {filename} (analysis JSON)...")
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
                dfs['pathways'] = pd.DataFrame(data) 
                print(f"Successfully created DataFrame for {filename}")
            elif 'results' in filename and filename.endswith('.json.gz'):
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    dfs['statistics'] = pd.DataFrame([data])
                elif isinstance(data, list): 
                    dfs['statistics'] = pd.DataFrame(data)
                else:
                    print(f"Warning: statistics file {filename} has an unexpected main data type: {type(data)}. Could not convert to DataFrame.")
            elif 'train' in filename and filename.endswith('.csv.gz'):
                dfs['train_stats'] = pd.read_csv(file_path, compression='gzip')
            elif 'test' in filename and filename.endswith('.csv.gz'):
                dfs['test_stats'] = pd.read_csv(file_path, compression='gzip')
            elif 'val' in filename and filename.endswith('.csv.gz'):
                dfs['val_stats'] = pd.read_csv(file_path, compression='gzip')
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
    return dfs['pathways'], dfs['statistics'], dfs['train_stats'], dfs['test_stats'], dfs['val_stats']

def analyze_statistics(statistics_df):
    row_data = statistics_df.iloc[0]
    train_metrics=row_data['train_metrics']
    test_metrics=row_data['test_metrics']
    val_metrics=row_data['val_metrics']

    metrics_list = [
        {'set': 'train', 'metrics': train_metrics},
        {'set': 'validation', 'metrics': val_metrics},
        {'set': 'test', 'metrics': test_metrics}
    ]
    
    metrics_df = pd.DataFrame([
        {'Set': 'Train', **train_metrics},
        {'Set': 'Validation', **val_metrics},
        {'Set': 'Test', **test_metrics}
    ])

    print("Metrics Summary Table:")
    print("=" * 80)
    print(metrics_df.to_string(index=False, float_format='%.4f'))
    print("=" * 80)

    print(f"\nAvailable metrics: {list(train_metrics.keys())}")
    
    return metrics_list, metrics_df


