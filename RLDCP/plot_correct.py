import matplotlib.pyplot as plt
import os

# 所有训练log.txt保存在txt下
filepath = "D:\\pythonProject\\plt\\txt\\"
# 所有绘制的correct折线图保存在plt下
savepath = "D:\\pythonProject\\plt\\fig\\"
dirs = os.listdir(filepath)
for file in dirs:
    f = open(filepath + file, 'r')
    lines = f.readlines()
    epoch = []
    global_correct = []
    average_row_correct = []
    a_correct = []
    b_correct = []
    c_correct = []
    for line in lines:
        if "[epoch:" in line:
            epoch_list = line.split(":")
            epoch_list = epoch_list[1].split("]")
            epoch.append(int(epoch_list[0]))
        if "global correct:" in line:
            global_correct_list = line.split(":")
            global_correct.append(float(global_correct_list[1]))
        if "average row correct:" in line:
            average_row_correct_list = line.split(":")
            average_row_correct_list = average_row_correct_list[1].split(",")
            a_correct_list = average_row_correct_list[0].split("'")
            a_correct.append(float(a_correct_list[1]))
            b_correct_list = average_row_correct_list[1].split("'")
            b_correct.append(float(b_correct_list[1]))
            c_correct_list = average_row_correct_list[2].split("'")
            c_correct.append(float(c_correct_list[1]))
    fig = plt.figure(figsize=(5, 5))
    new_epoch = []
    new_a_correct = []
    new_b_correct = []
    new_c_correct = []
    new_global_correct = []
    for i in epoch:
        # 设置间隔（每30个epoch画一个点）
        if i % 30 == 0:
            new_epoch.append(epoch[i])
            new_a_correct.append(a_correct[i])
            new_b_correct.append(b_correct[i])
            new_c_correct.append(c_correct[i])
            new_global_correct.append(global_correct[i])
    x = range(len(new_epoch))
    plt.plot(x, new_a_correct, marker='o', mec='r', mfc='w', label='a_correct')
    plt.plot(x, new_b_correct, marker='*', ms=10, label='b_correct')
    plt.plot(x, new_c_correct, marker='+', ms=10, label='c_correct')
    plt.plot(x, new_global_correct, marker='s', ms=10, label='global_correct')
    plt.legend()
    plt.xticks(x, new_epoch)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel("epoch", fontsize=10)
    plt.ylabel("correct", fontsize=10)
    plt.title("correct", fontsize=20)
    plt.savefig(savepath + file.split('.txt')[0] + '_correct.png')
# 我画错了应该画曲线图而非折线图，只是设置一下刻度就好了
