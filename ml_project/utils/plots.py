import matplotlib.pyplot as plt

def plot(train_loss, train_acc, test_loss, test_acc):
    # Create a new figure
    plt.figure()
    # Plot the training loss
    plt.plot(train_loss, label='Training Loss')
    # Plot the test loss
    plt.plot(test_loss, label='Test Loss')

    # Add a legend
    plt.legend()
    # Save the figure
    plt.savefig('graficos/loss.png')

    # Create a new figure
    plt.figure()
    # Plot the training accuracy
    plt.plot(train_acc, label='Training Accuracy')
    # Plot the test accuracy
    plt.plot(test_acc, label='Test Accuracy')

    # Add a legend
    plt.legend()
    # Save the figure
    plt.savefig('graficos/accuracy.png')