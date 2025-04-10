import argparse
import pandas as pd
from sklearn import metrics
from tqdm import tqdm
import numpy as np

def evaluate_results(log_file):
    # read the logits file
    pred = pd.read_csv(log_file)
    
    
    # Clean brackets
    pred['logits'] = pred['logits'].str.replace(r'[\[\]]', '', regex=True)
    
    # Convert string to array of floats
    logits_array = pred['logits'].apply(lambda x: np.fromstring(x, sep=' ') if isinstance(x, str) else np.array([np.nan, np.nan, np.nan]))
    
    # Split into separate columns
    pred[['pr_control', 'pr_stemi', 'pr_nstemi']] = pd.DataFrame(logits_array.tolist(), index=pred.index)
    
    # Combine MI predictions
    pred['prob_mi'] = pred['pr_stemi'] + pred['pr_nstemi']
    
    # Evaluate metrics - C-statistic / AUROC and pr
    roc_auc = metrics.roc_auc_score(pred['labels'], pred['prob_mi'])
    pr_auc = metrics.average_precision_score(pred['labels'], pred['prob_mi'])
    
    # Output
    tqdm.write(f"Results: control vs MI (STEMI+NSTEMI)")
    tqdm.write(f"ROC AUC: {roc_auc:.4f}")
    tqdm.write(f"PR AUC: {pr_auc:.4f}")    

# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--log_file", type=str, default="logs/logits.csv", help="path to logs")
    config, _ = parser.parse_known_args()
    
    evaluate_results(config.log_file)