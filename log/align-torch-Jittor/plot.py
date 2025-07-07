import pandas as pd
import matplotlib.pyplot as plt

# 文件夹路径
dataset = "dart"
jittor_folder = f"{dataset}/jittor"
torch_folder = f"{dataset}/torch"

pic_save_path = f"{dataset}/loss_{dataset}.png"

if __name__ == "__main__":
    # 读 CSV
    jittor_train = pd.read_csv(f"{jittor_folder}/train_log.csv")
    jittor_eval = pd.read_csv(f"{jittor_folder}/eval_log.csv")
    torch_train = pd.read_csv(f"{torch_folder}/train_log.csv")
    torch_eval = pd.read_csv(f"{torch_folder}/eval_log.csv")

    # 提取数据
    steps_train_jittor = jittor_train["step"]
    steps_eval_jittor = jittor_eval["step"]

    steps_train_torch = torch_train["step"]
    steps_eval_torch = torch_eval["step"]

    # 创建左右子图
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Torch 配色
    torch_main = "#4874CB"
    torch_valid = "#7BA1E0"

    # Jittor 配色
    jittor_main = "#EE822F"
    jittor_valid = "#F4A65B"

    # Torch 左子图
    axs[0].plot(steps_train_torch, torch_train["avg_loss"], label="Train Avg Loss", color=torch_main)
    axs[0].plot(steps_train_torch, torch_train["loss"], linestyle="--", color=torch_main, alpha=0.5, label="Train Loss")
    axs[0].plot(steps_eval_torch, torch_eval["valid_loss"], linestyle="-.", label="Eval Loss", color=torch_valid)

    axs[0].set_title(f"Torch loss in - {dataset}")
    axs[0].set_xlabel("Step")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    # Jittor 右子图
    axs[1].plot(steps_train_jittor, jittor_train["avg_loss"], label="Train Avg Loss", color=jittor_main)
    axs[1].plot(steps_train_jittor, jittor_train["loss"], linestyle="--", color=jittor_main, alpha=0.5, label="Train Loss")
    axs[1].plot(steps_eval_jittor, jittor_eval["valid_loss"], linestyle="-.", label="Eval Loss", color=jittor_valid)

    axs[1].set_title(f"Jittor loss - {dataset}")
    axs[1].set_xlabel("Step")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(pic_save_path, dpi=300)  # 可以改成 jpg/pdf/svg
    plt.show()
