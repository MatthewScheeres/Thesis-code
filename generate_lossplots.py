import os
import numpy as np
import json
import pandas as pd

count = 0

def print_json_contents(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        #print(json.dumps(data, indent=4))
        
def calculate_avg(data: list):
    avg_data = {}
    #print(data[0])
    if type(data[0]) == list:
        final_data = []
        for i in range(len(data[0])):
            final_data.append(calculate_avg([d[i] for d in data]))
        return final_data
    for key in data[0].keys():
        if isinstance(data[0][key], dict) and "confusion_matrix" not in key:
            # If the value is a dictionary, recursively calculate the average
            avg_data[key] = calculate_avg([d[key] for d in data])
        elif "confusion_matrix" not in key:
            # If the value is a number, calculate the average
            #print(key)
            avg_data[key] = sum(d[key] for d in data) / len(data)
    return avg_data
    
        

def search_and_print(directory, k=5):
    global count
    train_data = []
    eval_data = []
    total_data = []
    for root, dirs, files in os.walk(directory):
        if 'run_' in root:
            
            for file in files:
                if file == 'log_history.json' and 'fold_' in root:
                    with open(os.path.join(root, file), 'r') as f:
                        data = json.load(f)
                        
                    train_data.append(data["Train Logs"])
                    eval_data.append(data["Eval Logs"])
                    total_data.append(data["Total Logs"])
                    if len(train_data) == k:
                        train_data_avg = calculate_avg(train_data)
                        eval_data_avg = calculate_avg(eval_data)
                        total_data_avg = calculate_avg(total_data)
                        train_data, eval_data, total_data = [], [], []
                        # Reconstruction of the file
                        avg_data = {
                            "Train Logs": train_data_avg,
                            "Eval Logs": eval_data_avg,
                            "Total Logs": total_data_avg
                        }
                        save_dir = os.path.dirname(root)
                        
                        print(f"Saving to {save_dir}")
                        
                        with open(os.path.join(save_dir, 'average_statistics.json'), 'w') as f:
                            json.dump(avg_data, f, indent=4)
                        

search_and_print('.')
print (f"Total {count} files found")