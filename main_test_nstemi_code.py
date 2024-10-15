import argparse
from argparse import Namespace
import json
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np

from dataloader_code import MyECGDataset
from model import EnsembleECGModel
from eval_results import evaluate_results


if __name__ == "__main__":
    # system arugments
    sys_parser = argparse.ArgumentParser(add_help=False)
    sys_parser.add_argument("--input_data", type=str, default="data/out_mi.h5", help="path to data.",)
    sys_parser.add_argument("--log_dir", type=str, default="logs/", help="path to dir model weights")
    settings, _ = sys_parser.parse_known_args()

    # read config file
    file_path = os.path.join(os.getcwd(), settings.log_dir, "config.json")
    with open(file_path) as json_file:
        mydict = json.load(json_file)
    config = Namespace(**mydict)

    # Set device
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.write("Use device: {device:}\n".format(device=config.device))

    # -----------------------------------------------------------------------------
    # Dataloader
    # -----------------------------------------------------------------------------
    tqdm.write("Building data loaders...")
    dataset = MyECGDataset(settings.input_data)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    tqdm.write("Done!\n")

    # -----------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------
    tqdm.write("Define model...")
    model = EnsembleECGModel(config, settings.log_dir)
    tqdm.write("Done!\n")

    # -----------------------------------------------------------------------------
    # Test run
    # -----------------------------------------------------------------------------
    tqdm.write("Running test...")
    
    all_probs = []
    all_labels = []
    all_ids = []
    all_regs = []
    all_age = []
    
    pbar = tqdm(test_loader, total=len(test_loader), desc="Test")
    for batch_idx, batch in enumerate(pbar):
        # extract data from batch
        traces, labels, ids, regs, age, age_norm, sex = batch
        traces = traces.to(device=config.device)
        labels = labels.to(device=config.device)
        age_sex = torch.stack([sex, age_norm]).t().to(device=config.device)
        
        # forward pass
        with torch.no_grad():
            inp = traces, age_sex
            logits = model(inp)
            probs = F.softmax(logits, dim=-1)
            
        all_probs.append(probs.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_ids.append(ids)
        all_regs.append(regs)
        all_age.append(age)
        
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_ids = torch.cat(all_ids, dim=0)
    all_regs = torch.cat(all_regs, dim=0)
    all_age = torch.cat(all_age, dim=0)
    
    # save (ids, labels, logits) to csv file 
    df = pd.DataFrame({
        "ids": np.asarray(all_ids),
        "register_num": np.asarray(all_regs),
        "patient_age": np.asarray(all_age),
        "labels": np.asarray(all_labels),
        "logits": list(np.asarray(all_probs))
    })
    df.to_csv(os.path.join(settings.log_dir, "logits_code.csv"), index=False)
    tqdm.write("Logits saved to csv file.")
    
    # -----------------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------------
    tqdm.write("Evaluate results...")
    evaluate_results(os.path.join(settings.log_dir, "logits_code.csv"))
    tqdm.write("Done!\n")
    