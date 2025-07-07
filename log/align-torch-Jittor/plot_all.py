import pandas as pd
import matplotlib.pyplot as plt

# 文件夹路径
dataset = "dart"
jittor_folder = f"{dataset}/jittor"
torch_folder = f"{dataset}/torch"

pic_save_path = f"{dataset}/all_loss_{dataset}.png"

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

    # 画图
    plt.figure(figsize=(10, 6))

    # 颜色
    color_jittor = "#EE822F"
    color_torch = "#4874CB"

    # Jittor
    plt.plot(steps_train_jittor, jittor_train["avg_loss"], color=color_jittor, linestyle="-", label="Jittor Avg Train Loss")
    plt.plot(steps_train_jittor, jittor_train["loss"], color=color_jittor, linestyle="--", label="Jittor Train Loss")
    plt.plot(steps_eval_jittor, jittor_eval["valid_loss"], color=color_jittor, linestyle=":", label="Jittor Eval Loss")

    # Torch
    plt.plot(steps_train_torch, torch_train["avg_loss"], color=color_torch, linestyle="-", label="Torch Avg Train Loss")
    plt.plot(steps_train_torch, torch_train["loss"], color=color_torch, linestyle="--", label="Torch Train Loss")
    plt.plot(steps_eval_torch, torch_eval["valid_loss"], color=color_torch, linestyle=":", label="Torch Eval Loss")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Jittor vs Torch loss - {dataset}")
    plt.legend()
    plt.grid(True)

    # 先保存再展示
    plt.savefig(pic_save_path, dpi=300)
    plt.show()
