import matplotlib.pyplot as plt
import pickle
import os

def plot_training_history(history_path, output_dir="visualizations"):
    """
    Plots training and validation loss and accuracy from a history pickle file.
    """
    with open(history_path, 'rb') as file:
        history = pickle.load(file)

    os.makedirs(output_dir, exist_ok=True)

    epochs = range(1, len(history['loss']) + 1)

    # Plot Training and Validation Loss
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, history['loss'], 'r', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_validation_loss.png'))
    plt.close()

    # Plot Training and Validation Accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, history['accuracy'], 'r', label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_validation_accuracy.png'))
    plt.close()

    print(f"Training history plots saved to {output_dir}")

if __name__ == "__main__":
    history_file = "trained_model/training_history.pkl"
    if os.path.exists(history_file):
        plot_training_history(history_file)
    else:
        print(f"Error: Training history file not found at {history_file}.")
        print("Please run train_model_keras.py first to generate the history file.")
