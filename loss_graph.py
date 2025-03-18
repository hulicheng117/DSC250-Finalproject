import json
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

def extract_loss(event_file):
    # Load event file using TensorBoard's event accumulator
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload() 
    steps = ea.Scalars('loss') 
    eval_loss = ea.Scalars('eval_loss')  

    train_loss = [x.value for x in steps]
    eval_loss = [x.value for x in eval_loss]
    steps = [x.step for x in steps]

    return steps, train_loss, eval_loss

def plot_loss_curve(event_file, save_path="loss_curve.png"):
    steps, train_loss, eval_loss = extract_loss(event_file)

    plt.figure(figsize=(8, 5))
    plt.plot(steps, train_loss, label="Train Loss", marker="o")
    if eval_loss:
        plt.plot(steps[:len(eval_loss)], eval_loss, label="Validation Loss", marker="s")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid()

    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()


def main():
    plot_loss_curve("./logs_lora_finetune_r8/trainer_state.json", "./lora_r8_loss_curve.png")
    plot_loss_curve("./logs/trainer_state.json", "./full_loss_curve.png")
    plot_loss_curve("./logs_lora_finetune_r4/trainer_state.json", "./lora_r4_loss_curve.png")


if __name__ == "__main__":
    main()
