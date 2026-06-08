import matplotlib.pyplot as plt
import os

def plot_learning_curves(histories, labels, title, filename):
    plt.figure(figsize=(10, 6))
    
    for history, label in zip(histories, labels):
        test_errors = [err * 100 for err in history['test_error']] 
        plt.plot(test_errors, label=label)
        
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Test Error (%)')
    plt.legend()
    plt.grid(True)
    
    os.makedirs('results', exist_ok=True)
    plt.savefig(os.path.join('results', filename))
    plt.close()
