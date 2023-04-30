from model import *
from method import *
from config import *
from train import *

import random

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(SEED)
random.seed(SEED)


def set_env(config):
    num_ap_of_group1 = config['num_ap_of_group1'] if 'num_ap_of_group1' in config.keys(
    ) else NUM_AP_OF_GROUP1
    num_ap_of_group2 = config['num_ap_of_group2'] if 'num_ap_of_group2' in config.keys(
    ) else NUM_AP_OF_GROUP2
    num_md_of_group1 = config['num_md_of_group1'] if 'num_md_of_group1' in config.keys(
    ) else NUM_MD_OF_GROUP1
    num_md_of_group2 = config['num_md_of_group2'] if 'num_md_of_group2' in config.keys(
    ) else NUM_MD_OF_GROUP2
    m = config['M'] if 'M' in config.keys() else M
    n = config['N'] if 'N' in config.keys() else N
    num_md_of_group1 = config['num_md_of_group1'] if 'num_md_of_group1' in config.keys(
    ) else NUM_MD_OF_GROUP1
    num_md_of_group2 = config['num_md_of_group2'] if 'num_md_of_group2' in config.keys(
    ) else NUM_MD_OF_GROUP2
    devices = []
    aps = []
    # --------- 环境配置 ---------- #
    # 基站配置
    for i in range(m):
        if i < num_ap_of_group1:
            group = 0
            x = random.uniform(0, WIDTH)
            y = random.uniform(0, WIDTH)
            loc = [x, y]
            total_f = GROUP1_F
        else:
            group = 1
            x = random.uniform(0, WIDTH)
            y = random.uniform(0, WIDTH)
            loc = [x, y]
            total_f = GROUP2_F

        total_b = random.randint(B_MIN, B_MAX)
        ap = AP(id=i, loc=loc, total_b=total_b, total_f=total_f, group=group)
        aps.append(ap)

    if m == 2:
        ap_matrix = np.array([[0, 1], [1, 0]])
    elif m == 4:
        ap_matrix = np.array([[0, 1, 1, 0],
                              [1, 0, 0, 1],
                              [1, 0, 0, 1],
                              [0, 1, 1, 0]])
    else:
        ap_matrix = np.random.randint(1, 2, (m, m))
        for i in range(m):
            ap_matrix[i][(i+1) % m] = 0
            ap_matrix[(i+1) % m][i] = 0

    # 设备配置
    for i in range(n):
        if i < num_md_of_group1:
            group = 0
            x = random.uniform(0, WIDTH)
            y = random.uniform(0, WIDTH)
            loc = [x, y]
        else:
            group = 1
            x = random.uniform(0, WIDTH)
            y = random.uniform(0, WIDTH)
            loc = [x, y]

        f_c = round(random.uniform(F_C_MIN, F_C_MAX))
        f_p = random.uniform(F_P_MIN, F_P_MAX)
        input_size = round(random.uniform(INPUT_SIZE_MIN, INPUT_SIZE_MAX), 2)
        task = [input_size, random.randint(CP_MIN, CP_MAX) * input_size]
        device = Device(id=i, loc=loc, f_c=f_c,
                        f_p=f_p, task=task, group=group)
        devices.append(device)

    wired_width = [[0]*m for i in range(m)]  # 有线连接的带宽
    for i in range(m):
        for j in range(m):
            if wired_width[i][j] == 0 and wired_width[i][j] == 0:
                wired_width[i][j] = 100 * \
                    random.randint(WIRED_WIDTH_MIN, WIRED_WIDTH_MAX)
            else:
                wired_width[i][j] = wired_width[i][j]

    env = Env(N=n, M=m, devices=devices, aps=aps, ap_matrix=ap_matrix,
              wired_width=wired_width, path_loss=PATH_LOSS, noise=NOISE)
    # --------- 环境配置 ---------- #

    return env


def get_balance_result(ex_config):
    results = [[] for _ in range(len(ex_config['methods']))]
    for idx, method_name in enumerate(ex_config['methods']):
        for n in ex_config['num_md_of_group']:
            env_config = {
                'num_md_of_group1': n[0],
                'num_md_of_group2': n[1]
            }
            env = set_env(env_config)
            result = train(env, method_name, config=ex_config['method_config'])
            results[idx].append(result)
        plt.plot(ex_config['x'], results[idx], color=ex_config['colors'][idx],
                 marker=ex_config['markers'][idx], linewidth=2, label=method_name)
    with open('./data/balance_result', mode='w') as f:
        for i in range(len(results)):
            for j in range(len(results[i])):
                f.write('%.4f\t' % results[i][j])
            f.write('\n')
    # [TODO]标签待确认
    plt.xlabel('Unit Task Resource Ratio')
    plt.ylabel('Total delay')
    plt.legend()
    plt.savefig('./figs/balance_result.png', dpi=500)
    plt.show()


def get_delay_ap_result(ex_config, file_name=None):
    results = [[] for _ in range(len(ex_config['methods']))]
    for m in ex_config['M']:
        print('Num of APs:%d' % m)
        env_config = {
            'M': m,
            'num_ap_of_group1': m/2
        }
        env = set_env(env_config)
        for idx, method_name in enumerate(ex_config['methods']):
            e = copy.deepcopy(env)
            result = train(e, method_name, config=ex_config['method_config'])
            results[idx].append(result)
    with open('./data/delay_ap_result.txt', mode='w') as f:
        for i in range(len(results)):
            for j in range(len(results[i])):
                f.write('%.4f\t' % results[i][j])
            f.write('\n')
    draw_delay_ap_result(
        data=results, ex_config=ex_config, file_name=file_name)


def draw_delay_ap_result(ex_config, file_name, data=None):
    if not data:
        data = []
        with open(file=file_name, mode='r') as f:
            for line in f.readlines():
                data.append([float(i) for i in line.strip().split('\t')])
    for idx in range(len(data)):
        plt.plot(ex_config['M'], data[idx], color=ex_config['colors'][idx],
                 marker=ex_config['markers'][idx], linestyle=ex_config['linestyles'][idx], label=ex_config['methods'][idx])
    # [TODO]标签待确认
    plt.xlabel('Number of APs')
    plt.ylabel('Total delay')
    plt.legend()
    plt.savefig('./figs/delay_ap_result_MD%d.png' % N, dpi=500)
    plt.show()


if __name__ == "__main__":
    ex_name = EX_NAME
    ex_config = EX_CONFIG[ex_name]
    print('ex_name', ex_name)
    print('ex_config', ex_config)
    if ex_name == 'balance':
        get_balance_result(ex_config)
    if ex_name == 'delay_ap':
        # draw_delay_ap_result(ex_config, file_name='./data/delay_ap_result.txt')
        get_delay_ap_result(ex_config, file_name='./data/delay_ap_result%d.txt' % N)
