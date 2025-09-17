import pandas as pd
import matplotlib.pyplot as plt


# Plot training and validation loss curves
def plot_loss_curve(fruit, ax, i):
    # Load loss data
    loss_file = f'./{fruit}_losses.csv'
    df = pd.read_csv(loss_file)

    # Plot the graph on the specified axis (ax)
    ax.plot(df['Epoch'], df['Train_Loss'], label="Training Loss", color='green', marker='x')
    ax.tick_params(axis='x', direction='in', labelsize=20)  # y 轴刻度朝里
    ax.plot(df['Epoch'], df['Val_Loss'], label="Validation Loss", color='blue', marker='o')
    ax.tick_params(axis='y', direction='in', labelsize=20)  # y 轴刻度朝里
    # Set labels and title
    if i == 3 or i==4:
        ax.set_xlabel('Epochs', fontsize=20)
    if i == 0 or i == 3:
        ax.set_ylabel('Loss', fontsize=20)
    ax.set_title(f'{fruit} - Loss Curve', fontsize=20)

    # Configure axis ticks and grid
    ax.grid(True)
    # ax.legend()


# Main function
def main():
    fruits = ["SRR25180684", "ERR4706159", "SRR23347361", "DRR228524", "ERR9286493"]

    # Set global font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['text.usetex'] = True

    # Create a figure and an array of subplots (2 rows, 3 columns)
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))  # Create a grid for subplots

    # Flatten the array of axes for easier iteration
    axs = axs.flatten()

    # Plot each fruit's loss curve on the corresponding subplot
    for i, fruit in enumerate(fruits):
        plot_loss_curve(fruit, axs[i], i)

    # Remove the empty subplot (3rd column in 2nd row)
    fig.delaxes(axs[5])

    # Add a subplot for the legend
    axs[5].axis('off')  # Turn off the axes
    # Add the legend using bbox_to_anchor for precise positioning
    fig.legend(
        loc='upper center',
        labels=["Training Loss", "Validation Loss"],
        ncol=1,
        frameon=False,
        fontsize=20,  # Change font size here
        bbox_to_anchor=(0.85, 0.3)  # Adjust position (x, y)
    )

    # Adjust layout
    plt.tight_layout(pad=2.0)
    # Save the figure
    plt.savefig('./loss_curve.pdf', bbox_inches='tight')
    plt.show()  # Display the plot


if __name__ == "__main__":
    main()





