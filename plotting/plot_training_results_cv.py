import matplotlib.pyplot as plt
import json
import numpy as np
import os


def main():
    model_name = "MedRoBERTa.nl"
    model_dirname = "models--CLTL--MedRoBERTa.nl"
    model_type = "direct"
    symptom = "Koorts"
    timestamp = "20240429_2014"
    run_name = f"run_{symptom}_{model_dirname}_{timestamp}"
    results_dir = f'runs/{run_name}/'
    log_files = []

    for folder in os.listdir(results_dir):
        if os.path.isdir(os.path.join(results_dir, folder)):
            if folder.startswith('fold_'):
                log_file = json.load(open(os.path.join(results_dir, folder, 'log_history.json')))
                log_files.append(log_file)

    # Plot losses of different folds in separate plots
    for log_file in log_files:
        train_loss = [log['loss'] for log in log_file['Train Logs']]
        train_epochs = np.arange(0, 0.2*len(train_loss), 0.2)
        
        # Plot train loss
        plt.figure()
        plt.plot(train_epochs, train_loss, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
    plt.title(f'Train Loss of {model_name} on {symptom} per fold')
    plt.legend()
    plt.show()

    for log_file in log_files:
        eval_loss = [log['eval_loss'] for log in log_file['Eval Logs']]
        eval_epochs = np.arange(0, 0.2*len(eval_loss), 0.2)
        
        # Plot eval loss
        plt.figure()
        plt.plot(eval_epochs, eval_loss, label='Eval Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Eval Loss of {model_name} on {symptom} per fold')
    plt.legend()
    plt.show()

    # Calculate average train and eval loss
    avg_train_loss = np.mean([log['loss'] for log_file in log_files for log in log_file['Train Logs']])
    avg_eval_loss = np.mean([log['eval_loss'] for log_file in log_files for log in log_file['Eval Logs']])

    # Plot average train and eval loss
    plt.figure()
    plt.plot(avg_train_loss, label='Average Train Loss')
    plt.plot(avg_eval_loss, label='Average Eval Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Average Train and Eval Loss of {model_name} on {symptom}')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()