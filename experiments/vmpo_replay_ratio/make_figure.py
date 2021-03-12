import pandas
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from os import listdir
import numpy as np



def plot_with_std(x, y, label, color):
    std = np.std(y, axis=0)
    mean = np.mean(y, axis=0)
    plt.plot(x, mean, label=label, color=color)
    plt.fill_between(x, mean - 0.5 * std, mean + 0.5 * std, color=color, alpha=0.1)


path = Path(__file__).parent / '..' / 'logs' / 'vmpo_replay_ratio'
replay_ratio_1 = listdir(path / '1')
replay_ratio_2 = listdir(path / '2')
replay_ratio_4 = listdir(path / '4')
replay_ratio_8 = listdir(path / '8')

df_1 = [pandas.read_csv(path / '1' / run / 'progress.csv')[::2] for run in replay_ratio_1]
df_2 = [pandas.read_csv(path / '2' / run / 'progress.csv')[::2] for run in replay_ratio_2]
df_4 = [pandas.read_csv(path / '4' / run / 'progress.csv')[::2] for run in replay_ratio_4]
df_8 = [pandas.read_csv(path / '8' / run / 'progress.csv')[::2] for run in replay_ratio_8]

y_1 = np.stack([df['Traj_Infos/Return'] for df in df_1])
y_2 = np.stack([df['Traj_Infos/Return'] for df in df_2])
y_4 = np.stack([df['Traj_Infos/Return'] for df in df_4])
y_8 = np.stack([df['Traj_Infos/Return'] for df in df_8])

x = df_1[0]['Diagnostics/CumSteps']

with PdfPages(r'./vmpo_replay_ratio_experiment.pdf') as export_pdf:
    plot_with_std(x, y_1, 'Replay Ratio 1', 'red')
    plot_with_std(x, y_2, 'Replay Ratio 2', 'blue')
    plot_with_std(x, y_4, 'Replay Ratio 4', 'green')
    plot_with_std(x, y_8, 'Replay Ratio 8', 'darkorange')
    

    # plt.title('V-MPO with different replay ratios on Ant-v3', fontsize=14)
    plt.xlabel('Environment Steps', fontsize=14)
    plt.ylabel('Episode Reward', fontsize=14)
    plt.legend()

    plt.grid(True)
    # plt.show()
    export_pdf.savefig()
    plt.close()


