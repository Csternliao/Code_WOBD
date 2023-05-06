import random

from model import *
from config import *

random.seed(SEED)
np.random.seed(SEED)


def set_env(config):
    # --------- 环境配置 ---------- #
    n = config['n'] if 'n' in config.keys() else N
    m = config['m'] if 'm' in config.keys() else M
    aps = []
    devices = []
    # 基站配置
    for i in range(m):
        x = random.uniform(0, WIDTH)
        y = random.uniform(0, WIDTH)
        loc = [x, y]
        total_b = random.randint(B_MIN, B_MAX)
        total_f = random.randint(F_MIN, F_MAX)
        ap = AP(id=i, loc=loc, total_b=total_b, total_f=total_f)
        aps.append(ap)

    # if m == 4:
    #     ap_matrix = np.array([[0, 1, 1, 0],
    #                           [1, 0, 0, 1],
    #                           [1, 0, 0, 1],
    #                           [0, 1, 1, 0]])
    ap_matrix = np.random.randint(1, 2, (m, m))
    for i in range(m):
        ap_matrix[i][(i+1) % m] = 0
        ap_matrix[(i+1) % m][i] = 0

    # 设备配置
    for i in range(n):
        x = random.uniform(0, WIDTH)
        y = random.uniform(0, WIDTH)
        loc = [x, y]
        f_c = round(random.uniform(F_C_MIN, F_C_MAX))
        f_p = random.uniform(F_P_MIN, F_P_MAX)
        input_size = round(random.uniform(INPUT_SIZE_MIN, INPUT_SIZE_MAX), 2)
        task = [input_size, random.randint(CP_MIN, CP_MAX) * input_size]
        device = Device(id=i, loc=loc, f_c=f_c,
                        f_p=f_p, task=task)
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

    return env