import matplotlib.pyplot as plt
import numpy as np

def show_plot(y_label, model_name, symptom, x, y):
    plt.plot(x, y)
    plt.xlabel('Amount of Samples')
    plt.ylabel(y_label)
    plt.title(f"Plot of {y_label} for {model_name} with {symptom}")
    plt.grid(True)
    #plt.show()
    plt.savefig(f"plots/{y_label}_{model_name}_{symptom}.png")

def show_plots(metric, model_type, model_names, symptom, x, list_of_ys):
    for y, model_name in zip(list_of_ys, model_names):
        plt.plot(x, y)
    plt.legend(model_names)
    plt.xlabel('Amount of Samples')
    plt.ylabel(metric)
    plt.title(f"{str.capitalize(metric)} for {model_type} models on symptom: {symptom}")
    plt.grid(True)
    #plt.show()
    plt.savefig(f"plots/{metric}_{model_type}_{symptom}.png")


x = np.concatenate((np.arange(1, 11), np.arange(10, 901, 10)))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

precisions = [sigmoid((x - np.mean(x)) / np.std(x)) + np.random.rand(len(x)) * 0.05 for _ in range(4)]

show_plots("Precision", "prompt-based", ["MedRoBERTa.nl (pb)", "ChatDoctor", "RobBERT", "Dutch GPT-2"], "Cough", x, precisions)
show_plots("Precision", "classifier", ["MedRoBERTa.nl", "RobBERT", ], "Cough", x, precisions[:2])

show_plots("Recall", "prompt-based", ["MedRoBERTa.nl (pb)", "ChatDoctor", "RobBERT", "Dutch GPT-2"], "Cough", x, precisions)
show_plots("Recall", "classifier", ["MedRoBERTa.nl", "RobBERT", ], "Cough", x, precisions[:2])

show_plots("F1-score", "prompt-based", ["MedRoBERTa.nl (pb)", "ChatDoctor", "RobBERT", "Dutch GPT-2"], "Cough", x, precisions)
show_plots("F1-score", "classifier", ["MedRoBERTa.nl", "RobBERT", ], "Cough", x, precisions[:2])

x = np.concatenate((np.arange(1, 11), np.arange(10, 901, 10)))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

precisions = [sigmoid((x - np.mean(x)) / np.std(x)) + np.random.rand(len(x)) * 0.05 for _ in range(4)]


show_plots("Precision", "prompt-based", ["MedRoBERTa.nl (pb)", "ChatDoctor", "RobBERT", "Dutch GPT-2"], "Fever", x, precisions)
show_plots("Precision", "classifier", ["MedRoBERTa.nl", "RobBERT", ], "Fever", x, precisions[:2])

show_plots("Recall", "prompt-based", ["MedRoBERTa.nl (pb)", "ChatDoctor", "RobBERT", "Dutch GPT-2"], "Fever", x, precisions)
show_plots("Recall", "classifier", ["MedRoBERTa.nl", "RobBERT", ], "Fever", x, precisions[:2])

show_plots("F1-score", "prompt-based", ["MedRoBERTa.nl (pb)", "ChatDoctor", "RobBERT", "Dutch GPT-2"], "Fever", x, precisions)
show_plots("F1-score", "classifier", ["MedRoBERTa.nl", "RobBERT", ], "Fever", x, precisions[:2])

x = np.concatenate((np.arange(1, 11), np.arange(10, 901, 10)))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

precisions = [sigmoid((x - np.mean(x)) / np.std(x)) + np.random.rand(len(x)) * 0.05 for _ in range(4)]


show_plots("Precision", "prompt-based", ["MedRoBERTa.nl (pb)", "ChatDoctor", "RobBERT", "Dutch GPT-2"], "Shortness of breath", x, precisions)
show_plots("Precision", "classifier", ["MedRoBERTa.nl", "RobBERT", ], "Shortness of breath", x, precisions[:2])

show_plots("Recall", "prompt-based", ["MedRoBERTa.nl (pb)", "ChatDoctor", "RobBERT", "Dutch GPT-2"], "Shortness of breath", x, precisions)
show_plots("Recall", "classifier", ["MedRoBERTa.nl", "RobBERT", ], "Shortness of breath", x, precisions[:2])

show_plots("F1-score", "prompt-based", ["MedRoBERTa.nl (pb)", "ChatDoctor", "RobBERT", "Dutch GPT-2"], "Shortness of breath", x, precisions)
show_plots("F1-score", "classifier", ["MedRoBERTa.nl", "RobBERT", ], "Shortness of breath", x, precisions[:2])

