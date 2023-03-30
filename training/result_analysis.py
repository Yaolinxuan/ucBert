import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def  plot_loss_and_acc(log_history,output_dir):
    loss = []
    accuracy = []
    for log in log_history:
        if 'eval_loss' in log:
            loss.append(log['eval_loss'])
        if 'eval_accuracy' in log:
            accuracy.append(log['eval_accuracy'])
    print(loss)
    print(accuracy)
    #loss_history = trainer.state.log_history[:]["loss"]
    # 绘制 Loss 曲线
    plt.plot(loss)
    plt.xlabel('eval Steps')
    plt.ylabel('Loss')

    # 保存 Loss 曲线为 PNG 图片
    plt.title("loss history")
    plt.savefig(os.path.join(output_dir, "eval_roberta_loss.png"))
    plt.plot(accuracy)
    plt.xlabel('eval Steps')
    plt.ylabel('accuracy')

    # 保存 acc 曲线为 PNG 图片
    plt.title("accuracy history")
    plt.savefig(os.path.join(output_dir,
                "eval_roberta_accuracy.png"))