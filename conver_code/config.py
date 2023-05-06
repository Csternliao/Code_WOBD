# 方法参数
LAMDA = 0.1
THETA = 0.1
S_DELTA_WIN = 0.01
S_DELTA_LOSS = 0.05
CONVER_DIFF = 0.04
CONVER_EPOCH = 3

# 环境参数
SEED = 0
N = 40  # 设备数 10的倍数
M = 8  # AP数
MAX_EPOCHES = 300   # 最大迭代数

# 环境参数
WIDTH = 1  # 整片区域的长与宽，1km*1km
DIVIDE_WIDTH = 4  # 长和宽划分的份数，形成多个子区域
PATH_LOSS = 4  # 路径衰减指数
NOISE = 2e-13  # 高斯噪声值
WIRED_WIDTH_MIN = 1
WIRED_WIDTH_MAX = 10

# 设备参数
F_C_MIN = 1
F_C_MAX = 2.5
F_P_MIN = 0.05
F_P_MAX = 0.15
# AP参数
F_MIN = 20
F_MAX = 60
B_MIN = 5
B_MAX = 40

# 任务参数
INPUT_SIZE_MIN = 0.2
INPUT_SIZE_MAX = 4
CP_MIN = 1  # 每单位任务大小需要CPU多少转
CP_MAX = 5

METHOD_CONFIG = {
    'lamda': LAMDA,
    'theta': THETA,
    's_delta_loss': S_DELTA_LOSS,
    's_delta_win': S_DELTA_WIN,
    'conver_diff': CONVER_DIFF,
    'conver_epoch': CONVER_EPOCH,
    'max_epoches': MAX_EPOCHES
}

EX_NAME = 'cover_process'

EX_CONFIG = {
    'cover_process': {
        'methods': ['WoLF', 'Q-value'],
        'method_config': METHOD_CONFIG,
        'colors': ['cornflowerblue', 'orange'],
        'linestyles': ['-', '--'],
        'n': 20,
        'm': 5
        # 'markers': ['o', '^', 'P', '*', 's'],
        # 'num_md_of_group': [[16, 24], [23, 17], [27, 13], [29, 11], [30, 10]],
        # 'x': [1.00, 2.03, 3.12, 3.95, 4.50]
    },
    'cover_md': {
        'methods': ['WoLF', 'Q-value'],
        'method_config': METHOD_CONFIG,
        'colors': ['cornflowerblue', 'red'],
        'linestyles': ['-', '--'],
        'markers': ['o', '^'],
        'xlabel': 'Number of MUs',
        'ylabel': 'Convergence iterations',
        'file_name': './figs/conver_epoch_M_%d.pdf',
        'ns': [20, 30, 40, 50, 60],
        'm': 5
    }
}