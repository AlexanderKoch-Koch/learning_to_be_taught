import pandas
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import numpy as np

path = Path(__file__)

df_1_part_1 = pandas.read_csv(path.parent / '..' / 'logs' / 'ml45' / 'ml45_language_small_obs_new' / 'progress.csv')
df_1_part_2 = pandas.read_csv(path.parent / '..' / 'logs' / 'ml45' / 'ml45_language_part2_3' / 'progress.csv')
df_2_part_1 = pandas.read_csv(path.parent / '..' / 'logs' / 'ml45' / 'demonstrations' / 'run_0_part_0' / 'progress.csv')
df_2_part_2 = pandas.read_csv(path.parent / '..' / '..' / 'logs' / 'run_10' / 'progress.csv')

df_1_part_2['Diagnostics/CumSteps'] += df_1_part_1['Diagnostics/CumSteps'].values[-1] + 5e6
df_2_part_2['Diagnostics/CumSteps'] += df_2_part_1['Diagnostics/CumSteps'].values[-1] + 5e6
df_1 = pandas.concat([df_1_part_1, df_1_part_2])[:300:4]
df_2 = pandas.concat([df_2_part_1, df_2_part_2])[:300:4]

def smooth_values(x):
    x = np.concatenate(([x[0]], [x[0]], x, [x[-1]], [x[-1]]))
    x = x[:-4] + x[1:-3] + x[2:-2] + x[3:-1] + x[4:]
    x = x/5
    return x

with PdfPages(path.parent / 'ml45_experiment.pdf') as pdf:
    plt.plot(df_1['Diagnostics/CumSteps'], smooth_values(df_1['Traj_Infos/training_episode_success'].values),
             label='Language Instructions Train Tasks', color='red')
    plt.plot(df_2['Diagnostics/CumSteps'], smooth_values(df_2['Traj_Infos/training_episode_success'].values),
            label='Demonstrations Train Tasks', color='blue')

    plt.plot(df_1['Diagnostics/CumSteps'], smooth_values(df_1['Traj_Infos/testing_episode_success'].values), '--',
             label='Language Instructions Test Tasks', color='red')
    plt.plot(df_2['Diagnostics/CumSteps'], smooth_values(df_2['Traj_Infos/testing_episode_success'].values), '--',
            label='Demonstrations Test Tasks', color='blue')

    plt.xlabel('Environment Steps', fontsize=14)
    plt.ylabel('Sucess rate', fontsize=14)
    plt.legend()
    plt.grid(True)

    # plt.show()
    # plt.savefig(path.parent / 'ml45_experiment.pdf')
    pdf.savefig()
