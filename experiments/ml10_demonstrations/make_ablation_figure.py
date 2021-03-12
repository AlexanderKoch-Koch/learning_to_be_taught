import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from pathlib import Path
from os import listdir
import pandas


def smooth_values(x):
    x = np.concatenate(([x[0]], [x[0]], x, [x[-1]], [x[-1]]))
    x = x[:-4] + x[1:-3] + x[2:-2] + x[3:-1] + x[4:]
    x = x/5
    return x

if __name__ == '__main__':
    path = Path(__file__).parent / '..' / 'logs' / 'ml10' / 'ablation'
    demonstrations = listdir(path / '..' / 'demonstrations')

    df_normal = [pandas.read_csv(path / '..' / 'demonstrations' / run / 'progress.csv')[:80:1]
            for run in demonstrations]

    df_no_popart = pandas.read_csv(path / 'no_popart_0' / 'progress.csv')[:80:1]
    df_prev_action_obs = pandas.read_csv(path / 'demonstration_prev_action_obs' / 'progress.csv')[:80:1]
    df_full_time = pandas.read_csv(path / 'demonstrations_full_time_info' / 'progress.csv')[:80:1]

    y_normal = np.stack([df['Traj_Infos/training_episode_success'] for df in df_normal])
    y_no_popart = df_no_popart['Traj_Infos/training_episode_success'].values
    y_prev_action_obs = df_prev_action_obs['Traj_Infos/training_episode_success'].values
    y_full_time = df_full_time['Traj_Infos/training_episode_success'].values

    x = df_no_popart['Diagnostics/CumSteps']
    with PdfPages(r'./ablation_training_success.pdf') as export_pdf:
        plt.plot(x, smooth_values(np.mean(y_normal, axis=0)), label='normal')
        plt.plot(x, smooth_values(y_no_popart), label='No Pop-Art')
        plt.plot(x, smooth_values(y_prev_action_obs), label='Previous Action Observation')
        plt.plot(x, smooth_values(y_full_time), label='Full Time Observation')
        plt.xlabel('Environment Steps', fontsize=14)
        plt.ylabel('Average Trial Success', fontsize=14)
        plt.legend()
        plt.grid(True)
        # plt.show()
        export_pdf.savefig()
        plt.close()

    y_normal = np.stack([df['Traj_Infos/testing_episode_success'] for df in df_normal])
    y_no_popart = df_no_popart['Traj_Infos/testing_episode_success'].values
    y_prev_action_obs = df_prev_action_obs['Traj_Infos/testing_episode_success'].values
    y_full_time = df_full_time['Traj_Infos/testing_episode_success'].values

    with PdfPages(r'./ablation_test_success.pdf') as export_pdf:
        plt.plot(x, smooth_values(np.mean(y_normal, axis=0)), label='normal')
        plt.plot(x, smooth_values(y_no_popart), label='No Pop-Art')
        plt.plot(x, smooth_values(y_prev_action_obs), label='Previous Action Observation')
        plt.plot(x, smooth_values(y_full_time), label='Full Time Observation')
        plt.xlabel('Environment Steps', fontsize=14)
        plt.ylabel('Average Trial Success', fontsize=14)
        plt.legend()
        plt.grid(True)
        # plt.show()
        export_pdf.savefig()
        plt.close()
