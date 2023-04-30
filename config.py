# 方法参数
LAMDA = 0.1
THETA = 0.1
S_DELTA_WIN = 0.01
S_DELTA_LOSS = 0.05
CONVER_DIFF = 0.05
CONVER_EPOCH = 10

# 环境参数
SEED = 0
N = 40  # 设备数
M = 4  # AP数
# 下面两者相加应等于M
NUM_AP_OF_GROUP1 = M / 2
NUM_AP_OF_GROUP2 = M / 2
# 下面两者相加应等于N
NUM_MD_OF_GROUP1 = N * 3 / 8
NUM_MD_OF_GROUP2 = N * 5 / 8
MAX_EPOCHES = 500   # 最大迭代数

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
NUM_ACCESS_APS = 2

# AP参数
GROUP1_F = 50
GROUP2_F = 30
B_MIN = 5
B_MAX = 40


# 任务参数
INPUT_SIZE_MIN = 0.2
INPUT_SIZE_MAX = 4
CP_MIN = 1  # 每单位任务大小需要CPU多少转
CP_MAX = 5

# 实验设置
# balance   均衡性实验
# delay_ap  时延与AP数量关系
# delay_md  时延与MD数量关系
# conver    收敛性实验
EX_NAME = 'delay_ap'

METHOD_CONFIG = {
    'lamda': LAMDA,
    'theta': THETA,
    's_delta_loss': S_DELTA_LOSS,
    's_delta_win': S_DELTA_WIN,
    'conver_diff': CONVER_DIFF,
    'conver_epoch': CONVER_EPOCH,
    'max_epoches': MAX_EPOCHES
}

EX_CONFIG = {
    'balance': {
        'methods': ['WoLF', 'Q-value', 'GO', 'DO', 'RO'],
        'method_config': METHOD_CONFIG,
        'colors': ['b', 'orange', 'g', 'r', 'm'],
        'linestyles': ['-', '.', '--', '-.', '-'],
        'markers': ['o', '^', 'P', '*', 's'],
        'num_md_of_group': [[25, 15], [18, 22], [13, 27], [11, 29], [10, 30]],
        'x': [1.00, 2.04, 3.46, 4.39, 5.00]
    },
    'delay_ap': {
        'methods': ['WoLF', 'Q-value', 'GO', 'DO', 'RO'],
        'method_config': METHOD_CONFIG,
        'colors': ['b', 'orange', 'g', 'r', 'm'],
        'linestyles': ['-', '.', '--', '-.', '-'],
        'markers': ['o', '^', 'P', '*', 's'],
        'M': [4, 6, 8, 10, 12]
    },
    'delay_md': {

    },
    'conver': {

    }
}
