from model import *
from method import *
from config import *
from train import *

import random

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(SEED)
random.seed(SEED)


def get_all_env():
    all_devices = [[], []]
    all_aps = [[], []]
    # --------- 环境配置 ---------- #
    # 基站配置
    for i in range(MAX_APS):
        x = random.uniform(0, WIDTH)
        y = random.uniform(0, WIDTH)
        loc = [x, y]
        total_b = random.randint(B_MIN, B_MAX)
        if i < MAX_APS / 2:
            group = 0
            total_f = GROUP1_F
        else:
            group = 1
            total_f = GROUP2_F
        ap = AP(id=i, loc=loc, total_b=total_b, total_f=total_f, group=group)
        all_aps[group].append(ap)

    # if m == 4:
    #     ap_matrix = np.array([[0, 1, 1, 0],
    #                           [1, 0, 0, 1],
    #                           [1, 0, 0, 1],
    #                           [0, 1, 1, 0]])
    ap_matrix = np.random.randint(1, 2, (MAX_APS, MAX_APS))
    for i in range(MAX_APS):
        ap_matrix[i][(i+1) % MAX_APS] = 0
        ap_matrix[(i+1) % MAX_APS][i] = 0

    # 设备配置
    for i in range(MAX_DEVICES):
        x = random.uniform(0, WIDTH)
        y = random.uniform(0, WIDTH)
        loc = [x, y]

        f_c = round(random.uniform(F_C_MIN, F_C_MAX))
        f_p = random.uniform(F_P_MIN, F_P_MAX)
        input_size = round(random.uniform(INPUT_SIZE_MIN, INPUT_SIZE_MAX), 2)
        task = [input_size, random.randint(CP_MIN, CP_MAX) * input_size]
        if i < MAX_DEVICES * MD_GROUP_RATIO[0] / sum(MD_GROUP_RATIO):
            group = 0
        else:
            group = 1
        device = Device(id=i, loc=loc, f_c=f_c,
                        f_p=f_p, task=task, group=group)
        all_devices[group].append(device)

    wired_width = [[0]*MAX_APS for i in range(MAX_APS)]  # 有线连接的带宽
    for i in range(MAX_APS):
        for j in range(MAX_APS):
            if wired_width[i][j] == 0 and wired_width[i][j] == 0:
                wired_width[i][j] = 100 * \
                    random.randint(WIRED_WIDTH_MIN, WIRED_WIDTH_MAX)
            else:
                wired_width[i][j] = wired_width[i][j]

    return {
        'all_devices': all_devices,
        'all_aps': all_aps,
        'ap_matrix': ap_matrix,
        'wired_width': wired_width
    }


def set_env(config, all_env):
    m = config['M'] if 'M' in config.keys() else M
    n = config['N'] if 'N' in config.keys() else N
    num_ap_of_group1 = config['num_ap_of_group1'] if 'num_ap_of_group1' in config.keys() else m / 2
    num_ap_of_group2 = config['num_ap_of_group2'] if 'num_ap_of_group2' in config.keys() else m / 2
    num_md_of_group1 = config['num_md_of_group1'] if 'num_md_of_group1' in config.keys() else n * MD_GROUP_RATIO[0] / sum(MD_GROUP_RATIO)
    num_md_of_group2 = config['num_md_of_group2'] if 'num_md_of_group2' in config.keys() else n * MD_GROUP_RATIO[1] / sum(MD_GROUP_RATIO)

    all_devices = all_env['all_devices']
    all_aps = all_env['all_aps']
    ap_matrix = all_env['ap_matrix']
    wired_width = all_env['wired_width']

    all_devices = copy.deepcopy(all_devices)
    all_aps = copy.deepcopy(all_aps)

    devices = []
    for i in range(len(all_devices)):
        for j in range(len(all_devices[i])):
            if i == 0 and j < num_md_of_group1:
                devices.append(all_devices[i][j])
            elif i == 1 and j < num_md_of_group2:
                devices.append(all_devices[i][j])
            else:
                break
    for idx, device in enumerate(devices):
        device.id = idx

    aps = []
    aps_id = []
    for i in range(len(all_aps)):
        for j in range(len(all_aps[i])):
            if i == 0 and j < num_ap_of_group1:
                aps.append(all_aps[i][j])
            elif i == 1 and j < num_ap_of_group2:
                aps.append(all_aps[i][j])
            else:
                break
    for idx, ap in enumerate(aps):
        aps_id.append(ap.id)
        ap.id = idx

    ap_matrix = ap_matrix[aps_id][:, aps_id]
    wired_width = np.array(wired_width)[aps_id][:, aps_id].tolist()

    env = Env(N=n, M=m, devices=devices, aps=aps, ap_matrix=ap_matrix,
              wired_width=wired_width, path_loss=PATH_LOSS, noise=NOISE)
    # --------- 环境配置 ---------- #

    return env


def get_balance_result(ex_config, all_env):
    results = [[] for _ in range(len(ex_config['methods']))]
    avg_results = [[] for _ in range(len(ex_config['methods']))]
    for n in ex_config['num_md_of_group']:
        print("num_md_of_group1: %d, num_md_of_group2: %d" % (n[0], n[1]))
        env_config = {
            'num_md_of_group1': n[0],
            'num_md_of_group2': n[1]
        }
        env = set_env(env_config, all_env)
        for idx, method_name in enumerate(ex_config['methods']):
            e = copy.deepcopy(env)
            sum_result, avg_result = train(e, method_name, config=ex_config['method_config'])
            results[idx].append(sum_result)
            avg_results[idx].append(avg_result)
    with open('./data/balance_result.txt', mode='w') as f:
        for i in range(len(results)):
            for j in range(len(results[i])):
                f.write('%.4f\t' % results[i][j])
            f.write('\n')
    with open('./data/avg_balance_result.txt', mode='w') as f:
        for i in range(len(avg_results)):
            for j in range(len(avg_results[i])):
                f.write('%.4f\t' % avg_results[i][j])
            f.write('\n')
    draw_balance_result(
        data=results, ex_config=ex_config)
    # [TODO]标签待确认
    plt.xlabel('Unit Task Resource Ratio')
    plt.ylabel('Total delay')
    plt.legend()
    plt.savefig('./figs/balance_result.png', dpi=500)
    plt.show()


def draw_balance_result(ex_config, data=None, file_name=None):
    if not data and file_name:
        data = []
        with open(file=file_name, mode='r') as f:
            for line in f.readlines():
                data.append([float(i) for i in line.strip().split('\t')])
    elif data:
        pass
    else:
        raise ImportError
    for idx in range(len(data)):
        plt.plot(ex_config['x'], data[idx], color=ex_config['colors'][idx],
                 marker=ex_config['markers'][idx], linestyle=ex_config['linestyles'][idx], label=ex_config['methods'][idx])
    # [TODO]标签待确认
    plt.xlabel('Resource imbalance')
    plt.ylabel('Total delay')
    plt.grid(ls=':', color='gray')  # 设置网格
    plt.legend()
    plt.savefig('./figs/balance_result.png', dpi=500)
    plt.show()


def get_delay_ap_result(ex_config, all_env):
    results = [[] for _ in range(len(ex_config['methods']))]
    avg_results = [[] for _ in range(len(ex_config['methods']))]
    for m in ex_config['M']:
        print('Num of APs:%d' % m)
        env_config = {
            'M': m
        }
        env = set_env(env_config, all_env)
        for idx, method_name in enumerate(ex_config['methods']):
            e = copy.deepcopy(env)
            sum_result, avg_result = train(e, method_name, config=ex_config['method_config'])
            results[idx].append(sum_result)
            avg_results[idx].append(avg_result)
    with open('./data/delay_ap_result.txt', mode='w') as f:
        for i in range(len(results)):
            for j in range(len(results[i])):
                f.write('%.4f\t' % results[i][j])
            f.write('\n')
    with open('./data/avg_delay_ap_result.txt', mode='w') as f:
        for i in range(len(avg_results)):
            for j in range(len(avg_results[i])):
                f.write('%.4f\t' % avg_results[i][j])
            f.write('\n')
    draw_delay_ap_result(
        data=results, ex_config=ex_config)


def draw_delay_ap_result(ex_config, data=None, file_name=None):
    if not data and file_name:
        data = []
        with open(file=file_name, mode='r') as f:
            for line in f.readlines():
                data.append([float(i) for i in line.strip().split('\t')])
    elif data:
        pass
    else:
        raise ImportError
    for idx in range(len(data)):
        plt.plot(ex_config['M'], data[idx], color=ex_config['colors'][idx],
                 marker=ex_config['markers'][idx], linestyle=ex_config['linestyles'][idx], label=ex_config['methods'][idx])
    # [TODO]标签待确认
    plt.xlabel('Number of APs')
    plt.ylabel('Total delay')
    plt.grid(ls=':', color='gray')  # 设置网格
    plt.legend()
    plt.savefig('./figs/delay_ap_result.png', dpi=500)
    plt.show()


def get_delay_md_result(ex_config, all_env):
    results = [[] for _ in range(len(ex_config['methods']))]
    avg_results = [[] for _ in range(len(ex_config['methods']))]
    for n in ex_config['N']:
        print('Num of MDs:%d' % n)
        env_config = {
            'N': n
        }
        env = set_env(env_config, all_env)
        for idx, method_name in enumerate(ex_config['methods']):
            e = copy.deepcopy(env)
            sum_result, avg_result = train(e, method_name, config=ex_config['method_config'])
            results[idx].append(sum_result)
            avg_results[idx].append(avg_result)
    with open('./data/delay_md_result.txt', mode='w') as f:
        for i in range(len(results)):
            for j in range(len(results[i])):
                f.write('%.4f\t' % results[i][j])
            f.write('\n')
    with open('./data/avg_delay_md_result.txt', mode='w') as f:
        for i in range(len(avg_results)):
            for j in range(len(avg_results[i])):
                f.write('%.4f\t' % avg_results[i][j])
            f.write('\n')
    draw_delay_md_result(
        data=results, ex_config=ex_config)


def draw_delay_md_result(ex_config, data=None, file_name=None):
    if not data and file_name:
        data = []
        with open(file=file_name, mode='r') as f:
            for line in f.readlines():
                data.append([float(i) for i in line.strip().split('\t')])
    elif data:
        pass
    else:
        raise ImportError
    for idx in range(len(data)):
        plt.plot(ex_config['N'], data[idx], color=ex_config['colors'][idx],
                 marker=ex_config['markers'][idx], linestyle=ex_config['linestyles'][idx], label=ex_config['methods'][idx])
    # [TODO]标签待确认
    plt.xlabel('Number of MDs')
    plt.ylabel('Total delay')
    plt.grid(ls=':', color='gray')  # 设置网格
    plt.legend()
    plt.savefig('./figs/delay_md_result.png', dpi=500)
    plt.show()


if __name__ == "__main__":
    ex_name = EX_NAME
    ex_config = EX_CONFIG[ex_name]
    all_env = get_all_env()
    print('ex_name', ex_name)
    print('ex_config', ex_config)
    if ex_name == 'balance':
        get_balance_result(ex_config, all_env)
        # draw_balance_result(ex_config, file_name='./data/balance_result.txt')
    if ex_name == 'delay_ap':
        # draw_delay_ap_result(ex_config, file_name='./data/delay_ap_result40.txt')
        get_delay_ap_result(ex_config, all_env)
    if ex_name == 'delay_md':
        get_delay_md_result(ex_config, all_env)