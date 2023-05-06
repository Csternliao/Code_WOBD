'''
    对比方法版本
'''

import numpy as np
import random
import math
import pandas as pd
import trans_dispatch_process_delay
import copy
import matplotlib.pyplot as plt
import csv
import os
#from pylab import *

N = 40  # 设备数
M = 5  # AP数
width = 1  # 整片区域的长与宽，1km*1km
divide_width = 4  # 长和宽划分的份数，形成多个子区域
path_loss = 4  # 路径衰减指数
noise = 0.0000000000002  # 高斯噪声值
np.random.seed(0)
random.seed(0)
'''#记录每个AP的平均设备数和delay
average_MD_numbers_per_AP=0
average_delay_per_AP=0

#记录旧方法和对比方法每个AP的平均设备数和delay
average_MD_numbers_per_AP_old=0
average_delay_per_AP_old=0

average_MD_numbers_per_AP_GO=0
average_delay_per_AP_GO=0

average_MD_numbers_per_AP_FO=0
average_delay_per_AP_FO=0

average_MD_numbers_per_AP_DO=0
average_delay_per_AP_DO=0

average_MD_numbers_per_AP_RO=0
average_delay_per_AP_RO=0'''


simulation_numbers = 1
MD_iteration_numbers = 100


# ===================================
epsino_old = 0.05
epsino_wolf = 0.05
epsino_DO = 0.05

ave_round = 5

# Q-based算法收敛到epsino_old之内的*次total delay都计算出来
ave_round_old = ave_round
ave_round_wolf = ave_round
ave_round_DO = ave_round

fig_total_delay = []
fig_total_delay_old = []
fig_total_delay_DO = []

average_system_delay_wolf = 0
average_system_delay_old = 0

ave_iteration_wolf = 0
ave_iteration_old = 0
ave_iteration_DO = 0

# ===================================

# 记录方法造成的AP间最大时延者与平均AP时延的差距,以及卸载人数-----2020607
ave_max_delay_dif_wolf = 0
ave_offload_MD_num_wolf = 0

ave_max_delay_dif_old = 0
ave_offload_MD_num_old = 0

ave_max_delay_dif_GO = 0
ave_offload_MD_num_GO = 0

ave_max_delay_dif_FO = 0
ave_offload_MD_num_FO = 0

ave_max_delay_dif_DO = 0
ave_offload_MD_num_DO = 0

ave_max_delay_dif_RO = 0
ave_offload_MD_num_RO = 0

ave_APdelay_wolf = 0

ave_APdelay_old = 0

ave_APdelay_GO = 0

ave_APdelay_FO = 0

ave_APdelay_DO = 0

ave_APdelay_RO = 0

'''devices_policy_table=[]
device_y1=[[] for i in range(N)]
device_y2=[[] for i in range(N)]
device_y3=[[] for i in range(N)]
device_y4=[[] for i in range(N)]
device_y5=[[] for i in range(N)]
device_y6=[[] for i in range(N)]
device_y7=[[] for i in range(N)]
device_y8=[[] for i in range(N)]
device_y9=[[] for i in range(N)]
device_y10=[[] for i in range(N)]
device_y11=[[] for i in range(N)]
device_y12=[[] for i in range(N)]
device_y13=[[] for i in range(N)]
device_y14=[[] for i in range(N)]
device_y15=[[] for i in range(N)]
device_y16=[[] for i in range(N)]
device_y17=[[] for i in range(N)]
device_y18=[[] for i in range(N)]
device_y19=[[] for i in range(N)]
device_y10=[[] for i in range(N)]'''

'''devices_policy_table_old=[]
device_old_y1=[[] for i in range(N)]
device_old_y2=[[] for i in range(N)]
device_old_y3=[[] for i in range(N)]
device_old_y4=[[] for i in range(N)]
device_old_y5=[[] for i in range(N)]
device_old_y6=[[] for i in range(N)]
device_old_y7=[[] for i in range(N)]
device_old_y8=[[] for i in range(N)]
device_old_y9=[[] for i in range(N)]
device_old_y10=[[] for i in range(N)]
device_old_y11=[[] for i in range(N)]
device_old_y12=[[] for i in range(N)]
device_old_y13=[[] for i in range(N)]
device_old_y14=[[] for i in range(N)]
device_old_y15=[[] for i in range(N)]
device_old_y16=[[] for i in range(N)]
device_old_y17=[[] for i in range(N)]
device_old_y18=[[] for i in range(N)]
device_old_y19=[[] for i in range(N)]
device_old_y10=[[] for i in range(N)]'''

# 放收敛性比较用的新旧方法的延迟
'''x_index_poicy=[]
fig_total_delay=[]   
fig_total_delay_old=[]'''
'''#path_loss=[[[]]*M]*N
devices=[[[]]*6 for i in range(N)]
tasks=[[[]]*3 for i in range(N)]
APs=[[[]]*5 for i in range(M)]
dispatch_rate=[[[0]]*M for i in range(M)]
trans_rate=[[0]*M for i in range(N)]

wireless_weight=[[[0]]*M for i in range(N)]
CPU_weight=[[[0]*M for m in range(M)] for i in range(N)]
actual_rate=[[[0]]*M for i in range(N)]

devices_in_AP=[[] for i in range(M)]
tasks_in_AP=[[]for i in range(M)]
tasks_cpu_in_AP=[[]for i in range(M)]
tasks_input_in_AP=[[]for i in range(M)]
devices_dispatched=[[[]for i in range(M)] for i in range(M)]

trans_time=[[0]*M for i in range(N)]
dispatch_time=[0 for i in range(N)]
process_time=[0 for i in range(N)]'''

average_system_delay_wolf = 0
average_system_delay_old = 0
average_system_delay_GO = 0
average_system_delay_FO = 0
average_system_delay_DO = 0
average_system_delay_RO = 0

ave_simulation_delay_wolf = 0
ave_simulation_delay_old = 0
ave_simulation_delay_GO = 0
ave_simulation_delay_FO = 0
ave_simulation_delay_DO = 0
ave_simulation_delay_RO = 0

#########################----20200328----###########################
dispatch_road_width = [[0]*M for i in range(M)]  # 有线连接的带宽
for m in range(M):
    for n in range(M):
        if dispatch_road_width[m][n] == 0 and dispatch_road_width[n][m] == 0:
            dispatch_road_width[m][n] = 100*random.randint(1, 10)
        else:
            dispatch_road_width[m][n] = dispatch_road_width[n][m]

wired_weight = []  # [[[0]*M for m in range(M)] for i in range(N)]  #有线带宽分配
wired_weight_old = []
wired_weight_GO = []
wired_weight_FO = []
wired_weight_DO = []
wired_weight_RO = []

####################################################################

##############-----20200511-----####-----记录-----##############
s_delta_win = 0.0001
s_delta_lose = 0.1
####################################################################

for simualtion_index in range(simulation_numbers):
    # path_loss=[[[]]*M]*N
    devices = [[[]]*6 for i in range(N)]
    tasks = [[[]]*3 for i in range(N)]
    APs = [[[]]*5 for i in range(M)]
    dispatch_rate = [[[0]]*M for i in range(M)]
    trans_rate = [[0]*M for i in range(N)]

    wireless_weight = [[[0]]*M for i in range(N)]
    CPU_weight = [[[0]*M for m in range(M)] for i in range(N)]
    actual_rate = [[[0]]*M for i in range(N)]

    devices_in_AP = [[] for i in range(M)]
    tasks_in_AP = [[]for i in range(M)]
    tasks_cpu_in_AP = [[]for i in range(M)]
    tasks_input_in_AP = [[]for i in range(M)]
    devices_dispatched = [[[]for i in range(M)] for i in range(M)]

    trans_time = [[0]*M for i in range(N)]
    dispatch_time = [0 for i in range(N)]
    process_time = [0 for i in range(N)]

    # ----------------旧方法--------
    wireless_weight_old = [[[0]]*M for i in range(N)]
    CPU_weight_old = [[[0]*M for m in range(M)] for i in range(N)]
    actual_rate_old = [[[0]]*M for i in range(N)]

    devices_in_AP_old = [[] for i in range(M)]
    tasks_in_AP_old = [[]for i in range(M)]
    tasks_cpu_in_AP_old = [[]for i in range(M)]
    tasks_input_in_AP_old = [[]for i in range(M)]
    devices_dispatched_old = [[[]for i in range(M)] for i in range(M)]

    trans_time_old = [[0]*M for i in range(N)]
    dispatch_time_old = [0 for i in range(N)]
    process_time_old = [0 for i in range(N)]
    # -----------------------
    # ------------------------------------GO-------------------------------------------------
    #trans_rate_GO=[[0]*M for i in range(N)]
    wireless_weight_GO = [[[0]]*M for i in range(N)]
    CPU_weight_GO = [[[0]*M for m in range(M)] for i in range(N)]
    actual_rate_GO = [[[0]]*M for i in range(N)]

    devices_in_AP_GO = [[] for i in range(M)]
    tasks_in_AP_GO = [[]for i in range(M)]
    tasks_cpu_in_AP_GO = [[]for i in range(M)]
    tasks_input_in_AP_GO = [[]for i in range(M)]
    devices_dispatched_GO = [[[]for i in range(M)] for i in range(M)]

    trans_time_GO = [[0]*M for i in range(N)]
    dispatch_time_GO = [0 for i in range(N)]
    process_time_GO = [0 for i in range(N)]
    # ------------------------------------FO-------------------------------------------------------
    #trans_rate_FO=[[0]*M for i in range(N)]
    wireless_weight_FO = [[[0]]*M for i in range(N)]
    CPU_weight_FO = [[[0]*M for m in range(M)] for i in range(N)]
    actual_rate_FO = [[[0]]*M for i in range(N)]

    devices_in_AP_FO = [[] for i in range(M)]
    tasks_in_AP_FO = [[]for i in range(M)]
    tasks_cpu_in_AP_FO = [[]for i in range(M)]
    tasks_input_in_AP_FO = [[]for i in range(M)]
    devices_dispatched_FO = [[[]for i in range(M)] for i in range(M)]

    trans_time_FO = [[0]*M for i in range(N)]
    dispatch_time_FO = [0 for i in range(N)]
    process_time_FO = [0 for i in range(N)]
    # ------------------------------------DO-------------------------------------------------------
    #trans_rate_DO=[[0]*M for i in range(N)]
    wireless_weight_DO = [[[0]]*M for i in range(N)]
    CPU_weight_DO = [[[0]*M for m in range(M)] for i in range(N)]
    actual_rate_DO = [[[0]]*M for i in range(N)]

    devices_in_AP_DO = [[] for i in range(M)]
    tasks_in_AP_DO = [[]for i in range(M)]
    tasks_cpu_in_AP_DO = [[]for i in range(M)]
    tasks_input_in_AP_DO = [[]for i in range(M)]
    devices_dispatched_DO = [[[]for i in range(M)] for i in range(M)]

    trans_time_DO = [[0]*M for i in range(N)]
    dispatch_time_DO = [0 for i in range(N)]
    process_time_DO = [0 for i in range(N)]
    # ------------------------------------RO-------------------------------------------------------
    #trans_rate_RO=[[0]*M for i in range(N)]
    wireless_weight_RO = [[[0]]*M for i in range(N)]
    CPU_weight_RO = [[[0]*M for m in range(M)] for i in range(N)]
    actual_rate_RO = [[[0]]*M for i in range(N)]

    devices_in_AP_RO = [[] for i in range(M)]
    tasks_in_AP_RO = [[]for i in range(M)]
    tasks_cpu_in_AP_RO = [[]for i in range(M)]
    tasks_input_in_AP_RO = [[]for i in range(M)]
    devices_dispatched_RO = [[[]for i in range(M)] for i in range(M)]

    trans_time_RO = [[0]*M for i in range(N)]
    dispatch_time_RO = [0 for i in range(N)]
    process_time_RO = [0 for i in range(N)]

    # --------------------------------------
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    print("+++++++++++++++++++++++++++++++第", simualtion_index +
          1, "次实验开始+++++++++++++++++++++++++++++++")
    # ---------------------------20200807--------------------------------
    # Q-based算法收敛到epsino_old之内的*次total delay都计算出来
    ave_round_old = ave_round
    ave_round_wolf = ave_round
    ave_round_DO = ave_round

    # Q-based算法收敛到epsino_old之内的*次total delay平均值
    ave_round_delay_old = 0
    ave_round_delay_wolf = 0
    ave_round_delay_DO = 0

    # 用于记录
    print_nip_old = True
    print_nip_wolf = True
    print_nip_DO = True
    # ---------------------------20200807--------------------------------

    this_simu_ave_max_delay_dif_wolf = 0
    this_ave_offload_MD_num_wolf = 0

    this_simu_ave_max_delay_dif_old = 0
    this_ave_offload_MD_num_old = 0

    this_simu_ave_max_delay_dif_GO = 0
    this_ave_offload_MD_num_GO = 0

    this_simu_ave_max_delay_dif_FO = 0
    this_ave_offload_MD_num_FO = 0

    this_simu_ave_max_delay_dif_DO = 0
    this_ave_offload_MD_num_DO = 0

    this_simu_ave_max_delay_dif_RO = 0
    this_ave_offload_MD_num_RO = 0

    this_ave_APdelay_wolf = 0
    this_ave_APdelay_old = 0
    this_ave_APdelay_GO = 0
    this_ave_APdelay_FO = 0
    this_ave_APdelay_DO = 0
    this_ave_APdelay_RO = 0

    # 有关AP的设置
    for m in range(M):
        for n in range(M):
            dispatch_rate[m][n] = random.uniform(50, 100)
        APs[m][0] = [random.uniform(
            0, 1), random.uniform(0, 1)]  # 表示AP m的横纵坐标值
        x_index = int((APs[m][0][0]+width/divide_width)/(width/divide_width))
        y_index = int((APs[m][0][1]+width/divide_width)/(width/divide_width))
        # print(x_index,y_index)
        APs[m][1] = (y_index-1)*divide_width+x_index  # 表示AP m的所属区域
        APs[m][2].append(APs[m][1])  # AP m 覆盖的子区域
        if APs[m][1]-4 >= 0:
            APs[m][2].append(APs[m][1]-4)
        if APs[m][1]+4 <= (divide_width*divide_width):
            APs[m][2].append(APs[m][1]+4)
        if APs[m][1]-1 >= 0:
            APs[m][2].append(APs[m][1]-1)
        if APs[m][1]+1 <= (divide_width*divide_width):
            APs[m][2].append(APs[m][1]+1)
        APs[m][3] = 5+5*m  # AP m的总带宽
        APs[m][4] = 20+5*m  # AP m的总CPU频率
        # print("APs[",m,"]:",APs[m])
    # print("----APs:",APs)

    # 有关device的设置
    for i in range(N):
        devices[i][0] = i+M
        devices[i][1] = [random.uniform(
            0, 1), random.uniform(0, 1)]  # 表示device i的横纵坐标值
        x_index = int(
            (devices[i][1][0]+width/divide_width)/(width/divide_width))
        y_index = int(
            (devices[i][1][1]+width/divide_width)/(width/divide_width))
        devices[i][2] = (y_index-1)*divide_width+x_index  # 表示device i的所属区域
        devices[i][3] = round(random.uniform(1, 2.5))  # 表示device i的CPU频率
        devices[i][4] = random.uniform(0.05, 0.15)  # 表示device i的传输功率
        for m in range(M):
            if devices[i][2] in APs[m][2]:
                # device i 和 AP m的距离
                d_im = math.sqrt(
                    (devices[i][1][0]-APs[m][0][0])**2+(devices[i][1][1]-APs[m][0][1])**2)
                trans_rate[i][m] = APs[m][3] * \
                    math.log(1+(devices[i][4]*d_im **
                             (-path_loss))/noise)  # 传输速率
                devices[i][5].append(m)  # 表示设备可连接的AP序号
        tasks[i][0] = i+M
        tasks[i][1] = round(random.uniform(0.2, 4), 2)  # input size
        tasks[i][2] = random.randint(1, 5)*tasks[i][1]  # cpu cycles
        # print("Device[",i+M,"]:",devices[i])
        # print("Tasks[",i+M,"]:",tasks[i])
        # print("trans_rate[",i+M,"]:",trans_rate[i])
    # print("Device:",devices)
    # print("Tasks:",tasks)
    # print("trans_rate:",trans_rate)

    device_policy = [[]*(M+1) for i in range(N)]  # mixed policy of devices
    device_policy_old = [[]*(M+1) for i in range(N)]  # mixed policy of devices

    #====----=====----=======-____________________---=====----2020-7-17-----____________________-====------===-----=====--===#
    # 用于构建AP间联通情况，即拓扑结构 path_between_APs
    # wired_links用于表示任意两个AP之间有无有线连接
    #　设置随机种子，保证每次生成的随机数一样
    #rd = np.random.RandomState(888)
    # 随机整数
    AP_matrix = np.random.randint(1, 2, (M, M))
    for i in range(M):
        AP_matrix[i][(i+1) % M] = 0
        AP_matrix[(i+1) % M][i] = 0

    # shortest_path
    shortest_path = copy.deepcopy(AP_matrix)
    path_between_APs = [[[]for i in range(M)] for j in range(M)]
    # 记录每条有线电缆上的MD
    MDs_in_path_between_APs = [[[]for i in range(M)] for j in range(M)]

    for i in range(M):
        for j in range(M):
            if i == j:
                shortest_path[i][j] = 0
            if i != j and shortest_path[i][j] == 0:
                shortest_path[i][j] = 99999
            if i != j and shortest_path[i][j] == 1:
                path_between_APs[i][j].append(i)
                path_between_APs[i][j].append(j)

    for k in range(M):
        for i in range(M):
            for j in range(M):
                if shortest_path[i][j] > shortest_path[i][k] + shortest_path[k][j]:
                    shortest_path[i][j] = shortest_path[i][k] + \
                        shortest_path[k][j]
                    path_between_APs[i][j] = path_between_APs[i][k] + \
                        path_between_APs[k][j]
    for m in range(M):
        for n in range(M):
            path_between_APs[m][n] = list(set(path_between_APs[m][n]))
    # print(shortest_path)
    # print(path_between_APs)
    #---------------------------------------------------------------------------------#

    for i in range(N):
        totol_accessAP_number = len(set(devices[i][5]))
        probability = 1.0/(totol_accessAP_number+1)
        p_sum = 0
        # print('totol_accessAP_number','[',i+M,']:',totol_accessAP_number)
        device_policy[i].append(probability)
        p_sum = p_sum+probability  # device_policy[i][0]
        # print(p_sum)
        if totol_accessAP_number > 0:  # 如果MD可以连接到AP
            for m in range(M):
                # print(m)
                if m in devices[i][5]:
                    totol_accessAP_number = totol_accessAP_number-1
                    # print(totol_accessAP_number)
                    if totol_accessAP_number > 0:
                        device_policy[i].append(probability)
                        p_sum = p_sum+probability  # device_policy[i][m]
                        # print(p_sum)
                    else:
                        device_policy[i].append(
                            1-sum(device_policy[i]))  # (1-p_sum)
                else:
                    device_policy[i].append(0)
        else:
            for m in range(M):
                device_policy[i].append(0)
        # print("device_policy[", i+M, "]", device_policy[i])
    # ----------------合并新旧所用---------------------------
    device_policy_old = copy.deepcopy(device_policy)
    # print("device_policy",device_policy)
    # devices的Q-value初始值
    device_Q_value = [[0]*(M+1) for i in range(N)]
    device_Q_value_old = [[0]*(M+1) for i in range(N)]

    '''lamda=[0.2]*N   #更新策略时用
    theda=[0.1]*N   #更新Q值时用------5-24之前'''

    # ------5-24调整
    lamda = [0.1]*N  # 更新策略时用
    theda = [0.1]*N  # 更新Q值时用

    sum_Q_lamda_wolf = [0]*N  # 更新策略时用
    sum_Q_lamda = [0]*N  # 更新策略时用
    # devices 选择动作与更新策略、Q值，然后更新动作
    total_latency = [0]*N
    total_latency_old = [0]*N

    AP_latency = [0]*M
    AP_latency_old = [0]*M

    off_decision_index = [[] for i in range(N)]  # 统一卸载策略集合
    off_real_index = [[] for i in range(N)]  # 真实可选卸载策略集合

    Off_decision = [[] for i in range(N)]  # [[]]*N
    Off_decision_old = [[] for i in range(N)]  # [[]]*N

    converge_wolf = True  # 标记是否收敛！！！！！20200807！！！！！
    converge_old = True  # 标记是否收敛！！！！！20200807！！！！！
    converge_DO = True  # 标记是否收敛！！！！！20200807！！！！！

    notEndTime = True
    notEndTime_DO = True

    ################################
    # --------===========================-对比方法==========================---------
    device_policy_GO = copy.deepcopy(
        device_policy)  # 为 Greedy offloading 创建混合策略
    device_policy_FO = copy.deepcopy(device_policy)  # 为 Fast offloading 创建混合策略
    # 为 Direction offloading 创建混合策略
    device_policy_DO = copy.deepcopy(device_policy)
    device_policy_RO = copy.deepcopy(
        device_policy)  # 为 Random offloading 创建混合策略
    # print("device_policy",device_policy)
    # devices的Q-value初始值
    device_Q_value_GO = [[0]*(M+1) for i in range(N)]
    device_Q_value_FO = [[0]*(M+1) for i in range(N)]
    device_Q_value_DO = [[0]*(M+1) for i in range(N)]
    device_Q_value_RO = [[0]*(M+1) for i in range(N)]

    # 更新策略时用
    sum_Q_lamda_GO = [0]*N
    sum_Q_lamda_FO = [0]*N
    sum_Q_lamda_DO = [0]*N
    sum_Q_lamda_RO = [0]*N

    # devices 选择动作与更新策略、Q值，然后更新动作
    total_latency_GO = [0]*N
    total_latency_FO = [0]*N
    total_latency_DO = [0]*N
    total_latency_RO = [0]*N

    AP_latency_GO = [0]*M
    AP_latency_FO = [0]*M
    AP_latency_DO = [0]*M
    AP_latency_RO = [0]*M

    Off_decision_GO = [[] for i in range(N)]
    Off_decision_FO = [[] for i in range(N)]
    Off_decision_RO = [[] for i in range(N)]
    Off_decision_DO = [[] for i in range(N)]
    # --------===========================-对比方法部分设定结束==========================---------
    ##############-----20200511-----####-----记录-----##############
    # 博弈历史记录，用一个 N*（M+1）的列表
    gameHistory = [[1]*(M+1) for i in range(N)]

    # 平均策略，用一个 N*（M+1）的列表
    ave_policy = copy.deepcopy(device_policy)

    # 学习步长的参数
    s_delta = [0 for i in range(N)]

    sum_ave_value = [0 for i in range(N)]
    sum_current_value = [0 for i in range(N)]
    ################################################################

    for i in range(N):
        off_decision_index[i].append(i+M)
        off_real_index[i].append(i+M)
        for m in range(M):
            off_decision_index[i].append(m)
            if m in devices[i][5]:
                off_real_index[i].append(m)
        # print('off_decision_index[', i+M, ']:', off_decision_index[i])
        # print('off_real_index[', i+M, ']:', off_real_index[i])
    # =========================--------------  MD迭代   -----------------------===========================
    leiji_total_latency_wolf = []
    leiji_total_latency_old = []
    leiji_total_latency_GO = []
    leiji_total_latency_FO = []
    leiji_total_latency_DO = []
    leiji_total_latency_RO = []

    for t_d in range(MD_iteration_numbers):  # device 迭代次数
        if t_d % 5 == 0:
            print("t_d", t_d)
        # print("t_d",t_d)
        # ---------------------记录上一时间片的MDs_policy------------------------
        MDs_policy_old = copy.deepcopy(device_policy_old)
        MDs_policy_wolf = copy.deepcopy(device_policy)
        MDs_policy_DO = copy.deepcopy(device_policy_DO)
        # -------------------------------------------------------------------------

        '''if simualtion_index==0:
            devices_policy_table.append(device_policy)
            x_index_poicy.append(t_d)'''
        local_MD = []
        local_MD_old = []
        local_MD_GO = []
        local_MD_FO = []
        local_MD_DO = []
        local_MD_RO = []

        dispatch_decision = [[] for i in range(N)]
        dispatch_decision_old = [[] for i in range(N)]
        dispatch_decision_GO = [[] for i in range(N)]
        dispatch_decision_FO = [[] for i in range(N)]
        dispatch_decision_DO = [[] for i in range(N)]
        dispatch_decision_RO = [[] for i in range(N)]

        for m in range(M):
            devices_in_AP[m].clear()
            tasks_input_in_AP[m].clear()
            tasks_cpu_in_AP[m].clear()
            #
            devices_in_AP_old[m].clear()
            tasks_input_in_AP_old[m].clear()
            tasks_cpu_in_AP_old[m].clear()
            #
            devices_in_AP_GO[m].clear()
            tasks_input_in_AP_GO[m].clear()
            tasks_cpu_in_AP_GO[m].clear()

            devices_in_AP_FO[m].clear()
            tasks_input_in_AP_FO[m].clear()
            tasks_cpu_in_AP_FO[m].clear()

            devices_in_AP_DO[m].clear()
            tasks_input_in_AP_DO[m].clear()
            tasks_cpu_in_AP_DO[m].clear()

            devices_in_AP_FO[m].clear()
            tasks_input_in_AP_FO[m].clear()
            tasks_cpu_in_AP_FO[m].clear()

            devices_in_AP_RO[m].clear()
            tasks_input_in_AP_RO[m].clear()
            tasks_cpu_in_AP_RO[m].clear()

        if converge_wolf and ave_round_wolf > 0 and notEndTime:  # and print_nip_wolf:202027282043改
            # ======================================下面  实现  WoLF-PHC  ============================================================
            for i in range(N):
                # 记录MD2的策略
                '''if simualtion_index==0:
                    device_y1[i].append(device_policy[i][0])
                    device_y2[i].append(device_policy[i][1])
                    device_y3[i].append(device_policy[i][2])
                    device_y4[i].append(device_policy[i][3])
                    device_y5[i].append(device_policy[i][4])
                    device_y6[i].append(device_policy[i][5])'''

                dispatch_time[i] = 0
                process_time[i] = 0

                for m in range(M):
                    trans_time[i][m] = 0
                local_MD.append(i+M)
                # device根据策略选动作#,p=device_policy[i].ravel()
                Off_decision[i] = np.random.choice(
                    off_decision_index[i], p=device_policy[i])

                # print(i+M,"offloading_decision:",Off_decision[i])
                if Off_decision[i] < M:
                    # --------2020-5-11-----记录动作选择的的历史次数----------------
                    gameHistory[i][Off_decision[i]+1] += 1
                    # ---------------------------------------------------
                    dispatch_decision[i] = Off_decision[i]
                    local_MD.remove(i+M)
                    devices_in_AP[Off_decision[i]].append(i+M)  # 选择卸载到某AP的设备号
                    tasks_input_in_AP[Off_decision[i]].append(
                        tasks[i][1])  # 选择卸载到某AP的任务inputsize
                    tasks_cpu_in_AP[Off_decision[i]].append(
                        tasks[i][2])  # 选择卸载到某AP的任务cpucycles
                else:
                    # --------2020-5-11-----记录动作选择的的历史次数----------------
                    gameHistory[i][0] += 1

                # --------------------------平均策略更新---------------------------------
                # for a in range(M+1):
                    # ave_policy[i][a]=ave_policy[i][a]+(device_policy[i][a]-ave_policy[i][a])/gameHistory[i][a]#(t_d+1)
                    # ave_policy[i][a]=
                sum_Q_lamda_wolf[i] = 0
                sum_Q_lamda_wolf[i] = math.exp(device_Q_value[i][0]/lamda[i])
                sum_policy = 0.0
                ave_policy[i].clear()
                ave_policy[i].append(0)
                for m in range(M):
                    # print(math.exp(device_Q_value[i][m+1]/lamda[i]))
                    try:
                        tmp = math.exp(device_Q_value[i][m+1]/lamda[i])
                    except OverflowError:
                        tmp = float('inf')
                    sum_Q_lamda_wolf[i] = sum_Q_lamda_wolf[i]+tmp
                # print("sum_Q_lamda[",i+M,"]",sum_Q_lamda[i])
                for m in range(M):
                    if m in devices[i][5]:
                        x = (
                            math.exp(device_Q_value[i][m+1]/lamda[i]))/sum_Q_lamda_wolf[i]
                        ave_policy[i].append(x)
                        sum_policy = sum_policy+x
                    else:
                        ave_policy[i].append(0)
                    #print("sum_policy of",i+M,":",sum_policy)
                ave_policy[i][0] = 1-sum_policy

                sum_thisAve = sum(ave_policy[i])
                # ----------------------5-19新增-----------------------------
                for a in range(M+1):
                    # 平均策略归一化处理
                    ave_policy[i][a] = ave_policy[i][a]/sum_thisAve
                # ---------------------------------------------------
            # print("devices_in_AP",devices_in_AP)
            # print('off_decision:',Off_decision)
            for m in range(M):
                for n in range(M):
                    devices_dispatched[m][n].clear()
                devices_dispatched[m][m] = copy.deepcopy(devices_in_AP[m])
            trans_dispatch_process_delay.s_delay(
                N, M, devices, APs, tasks, devices_in_AP, trans_rate, trans_time, wireless_weight, actual_rate)
            trans_dispatch_process_delay.c_delay(
                M, N, APs, devices, tasks, devices_dispatched, process_time, CPU_weight)
            for i in local_MD:  # 本地执行的设备的时延存储
                total_latency[i-M] = process_time[i-M]
            # ------------------------------------------------------------------------------------------
            # ————————————————————————2020-7-17——————更新每个电缆上的MD—————————————————————————————————
            MDs_in_path_between_APs = [[[]for i in range(M)]for j in range(M)]
            for m in range(M):
                for n in range(M):
                    # print(m,n,'----',path_between_APs[m][n])
                    # print(devices_dispatched[m][n])
                    for i in devices_dispatched[m][n]:
                        for m_ in range(len(path_between_APs[m][n])-1):
                            if m_ != len(path_between_APs[m][n])-1:
                                MDs_in_path_between_APs[path_between_APs[m][n]
                                                        [m_]][path_between_APs[m][n][m_+1]].append(i)
            # print('--',MDs_in_path_between_APs)
            # —————————————————————————————————————————————————————————————————————
            trans_dispatch_process_delay.d_delay(N, M, tasks, devices_dispatched, dispatch_time,
                                                 dispatch_road_width, wired_weight, path_between_APs, MDs_in_path_between_APs)
            # ------------------------------------------------------------------------------------------
            for m in range(M):  # 初始dispatch决定带来的时延
                for i in devices_in_AP[m]:
                    total_latency[i-M] = trans_time[i-M][m] + \
                        dispatch_time[i-M]+process_time[i-M]
                    AP_latency[m] = AP_latency[m]+total_latency[i-M]
            # print("AP初始收到的任务:",devices_dispatched)
            # # -----------------------------AP迭代------------------------------------------------------------------------------------------
            # # 最开始的初始化dispatch decisions就是AP自己执行自己的，然后如果有AP能通过分派任务使得自身的任务时延减小，令这些AP中 \sum(CPUcyccles)/\sum(inputsiza)最大的AP先更新策略,比例要不要加期望要再想一下
            # # \sum(CPUcyccles)/\sum(inputsiza)的存储列表
            # tasks_input_cpu_ratio_rank = [[]for i in range(M)]

            # # ============================开始AP的决策更新=(把对比方法的也同时算出来)=====================================
            # for m in range(M):  # 计算并降序排序\sum(CPUcyccles)/\sum(inputsiza)
            #     if sum(tasks_input_in_AP[m]) > 0:
            #         tasks_input_cpu_ratio_rank[m] = (
            #             sum(tasks_cpu_in_AP[m])/APs[m][4])  # sum(tasks_input_in_AP[m])
            #     else:
            #         tasks_input_cpu_ratio_rank[m] = 0
            #     # tasks_input_cpu_ratio_rank_copy.append(sum(tasks_cpu_in_AP[m])/sum(tasks_input_in_AP[m]))

            #     # ---------------------------------------------------------------------------------------------
            # # print(tasks_input_cpu_ratio_rank)
            # tasks_input_cpu_ratio_rank_copy = np.argsort(
            #     -np.array(tasks_input_cpu_ratio_rank))

            # # print(tasks_input_cpu_ratio_rank_copy)
            # # print("tasks_cpu_in_AP", tasks_cpu_in_AP)
            # # print("devices_dispatched", devices_dispatched)
            # for t_A in range(1):
            #     # print("t_A",t_A)
            #     while(True):  # 只要AP策略没收敛，就一直更新
            #         devices_dispatched_compare_copy = copy.deepcopy(
            #             devices_dispatched)
            #         # print("devices_dispatched_compare_copy:",devices_dispatched_compare_copy)
            #         for h in range(M):  # 按排序使AP异步更新
            #             # m为\sum(CPUcyccles)/\sum(inputsiza)第h大的AP
            #             m = tasks_input_cpu_ratio_rank_copy[h]
            #             if tasks_input_cpu_ratio_rank[m] == 0:
            #                 break
            #             # print("AP",m,"在更新")
            #             # print("tasks_cpu_in_AP[",m,"]",tasks_cpu_in_AP[m])
            #             # task_cpu_in_AP_copy=copy.deepcopy(tasks_cpu_in_AP[m])
            #             # task_cpu_in_AP_copy是降序排序的原列表从大到小的索引
            #             task_cpu_in_AP_copy = np.argsort(
            #                 -np.array(tasks_cpu_in_AP[m]))
            #             # print(task_cpu_in_AP_copy)

            #             # devices_dispatched_compare_copy=copy.deepcopy(devices_dispatched)
            #             # print("devices_dispatched_compare_copy in h=",h,":",devices_dispatched_compare_copy)
            #             for j in range(len(set(devices_in_AP[m]))):
            #                 # print(task_cpu_in_AP_copy[j])
            #                 max_i_cpu = task_cpu_in_AP_copy[j]
            #                 # print(max_i_cpu)
            #                 # 求出m上cpu cycles最大的任务所属设备号
            #                 max_i = devices_in_AP[m][max_i_cpu]
            #                 # print(max_i)
            #                 # latency_maxi_copy=copy.deepcopy(total_latency[max_i-M]) #把max_i原始时延先拷贝下来
            #                 # 下面这一顿deepcopy是为了存储到最好的分派决策之后再更新原始的策略-----
            #                 dispatch_time_copy = copy.deepcopy(dispatch_time)
            #                 process_time_copy = copy.deepcopy(process_time)
            #                 CPU_weight_copy = copy.deepcopy(CPU_weight)
            #                 total_latency_copy = copy.deepcopy(total_latency)
            #                 AP_latency_copy = copy.deepcopy(AP_latency)
            #                 # Off_decision_copy=copy.deepcopy(Off_decision)
            #                 dispatch_decision_copy = copy.deepcopy(
            #                     dispatch_decision)
            #                 devices_dispatched_copy = copy.deepcopy(
            #                     devices_dispatched)  # -------------
            #                 MDs_in_path_between_APs_copy = copy.deepcopy(
            #                     MDs_in_path_between_APs)
            #                 for n in range(M):  # 循环，为max_i挑出最好的修改分派方案 and n!=m
            #                     # 下面这一顿deepcopy是使得每次计算针对当前max_i的分派决定需要有好效果才更新到原有的copy版本-----
            #                     dispatch_time_copy2 = copy.deepcopy(
            #                         dispatch_time_copy)
            #                     process_time_copy2 = copy.deepcopy(
            #                         process_time_copy)
            #                     CPU_weight_copy2 = copy.deepcopy(
            #                         CPU_weight_copy)
            #                     total_latency_copy2 = copy.deepcopy(
            #                         total_latency_copy)
            #                     AP_latency_copy2 = copy.deepcopy(
            #                         AP_latency_copy)
            #                     # Off_decision_copy2=copy.deepcopy(Off_decision_copy)
            #                     dispatch_decision_copy2 = copy.deepcopy(
            #                         dispatch_decision_copy)
            #                     MDs_in_path_between_APs_copy2 = copy.deepcopy(
            #                         MDs_in_path_between_APs_copy)

            #                     devices_dispatched_copy2 = copy.deepcopy(
            #                         devices_dispatched_copy)  # -------------

            #                     # 把cpucycles最大的设备先从原分配方案中删去，然后换新的计算延迟
            #                     devices_dispatched_copy2[m][dispatch_decision_copy2[max_i-M]].remove(
            #                         max_i)
            #                     # Off_decision_copy2[max_i-M]=n
            #                     # ————————————————————————2020-7-17——————虚拟移除原来分派选路上的MD——————————
            #                     # print(m,dispatch_decision_copy2[max_i-M],'-=-',n)
            #                     # print(path_between_APs[m][dispatch_decision_copy2[max_i-M]])
            #                     for m_ in range(len(path_between_APs[m][dispatch_decision_copy2[max_i-M]])-1):
            #                         if m_ != len(path_between_APs[m][dispatch_decision_copy2[max_i-M]])-1:
            #                             # print(m_,path_between_APs[m][dispatch_decision_copy2[max_i-M]][m_],path_between_APs[m][dispatch_decision_copy2[max_i-M]][m_+1],MDs_in_path_between_APs_copy2[path_between_APs[m][dispatch_decision_copy2[max_i-M]][m_]][path_between_APs[m][dispatch_decision_copy2[max_i-M]][m_+1]],max_i)
            #                             MDs_in_path_between_APs_copy2[path_between_APs[m][dispatch_decision_copy2[max_i-M]]
            #                                                           [m_]][path_between_APs[m][dispatch_decision_copy2[max_i-M]][m_+1]].remove(max_i)
            #                     # —————————————————————————————————————————————————————————————————————————
            #                     dispatch_decision_copy2[max_i-M] = n
            #                     # 接下来计算把max_i放到n上产生的时延
            #                     devices_dispatched_copy2[m][dispatch_decision_copy2[max_i-M]].append(
            #                         max_i)
            #                     # trans_dispatch_process_delay.d_delay(max_i,M,tasks,devices_dispatched_copy2,dispatch_time_copy2,dispatch_rate)
            #                     # ————————————————————————2020-7-17——————虚拟更新当前选择分派路上的MD—————————————————————————————————
            #                     for m_ in range(len(path_between_APs[m][n])-1):
            #                         if m_ != len(path_between_APs[m][dispatch_decision_copy2[max_i-M]])-1:
            #                             MDs_in_path_between_APs_copy2[path_between_APs[m][dispatch_decision_copy2[max_i-M]]
            #                                                           [m_]][path_between_APs[m][dispatch_decision_copy2[max_i-M]][m_+1]].append(max_i)
            #                     # ——————————————————————————————————————————————————————————————————————
            #                     trans_dispatch_process_delay.d_delay(
            #                         N, M, tasks, devices_dispatched_copy2, dispatch_time, dispatch_road_width, wired_weight, path_between_APs, MDs_in_path_between_APs_copy2)
            #                     trans_dispatch_process_delay.c_delay(
            #                         M, N, APs, devices, tasks, devices_dispatched_copy2, process_time_copy2, CPU_weight_copy2)
            #                     AP_latency_try = 0
            #                     devices_not_in = []
            #                     for i in range(N):
            #                         devices_not_in.append(i+M)
            #                     for i in devices_in_AP[m]:
            #                         devices_not_in.remove(i)
            #                         #print("----------before copy----------")
            #                         # print("total_latency_copy2[",i,"]",total_latency_copy2[i-M])
            #                         total_latency_copy2[i-M] = trans_time[i-M][m] + \
            #                             dispatch_time_copy2[i-M] + \
            #                             process_time_copy2[i-M]
            #                         #print("----------after copy----------")
            #                         # print("total_latency_copy2[",i,"]",total_latency_copy2[i-M])
            #                         # Ap_lateny_copy[m]=0
            #                         AP_latency_try = AP_latency_try + \
            #                             total_latency_copy2[i-M]
            #                         # AP_latency_copy[m]=AP_latency_copy[m]+total_latency_copy[i]
            #                     # print("AP",m,"_latency_try:",AP_latency_try)
            #                     # print("AP",m,"_latency_copy2:",(AP_latency_copy2[m]))
            #                     for i in devices_not_in:
            #                         total_latency_copy2[i-M] = sum(
            #                             trans_time[i-M])+dispatch_time_copy2[i-M]+process_time_copy2[i-M]
            #                     # 如果这个选择是比原来的好，那就更新策略，保存时延结果
            #                     if AP_latency_try < AP_latency_copy2[m]:
            #                         AP_latency_copy[m] = copy.deepcopy(
            #                             AP_latency_try)
            #                         dispatch_time_copy = copy.deepcopy(
            #                             dispatch_time_copy2)
            #                         process_time_copy = copy.deepcopy(
            #                             process_time_copy2)
            #                         CPU_weight_copy = copy.deepcopy(
            #                             CPU_weight_copy2)
            #                         total_latency_copy = copy.deepcopy(
            #                             total_latency_copy2)
            #                         # Off_decision_copy=copy.deepcopy(Off_decision_copy2)
            #                         dispatch_decision_copy = copy.deepcopy(
            #                             dispatch_decision_copy2)
            #                         devices_dispatched_copy = copy.deepcopy(
            #                             devices_dispatched_copy2)
            #                         MDs_in_path_between_APs_copy = copy.deepcopy(
            #                             MDs_in_path_between_APs_copy2)

            #                 # 下面的deepcopy把针对与当前max_i的最好结果进行更新
            #                 AP_latency[m] = copy.deepcopy(AP_latency_copy[m])
            #                 dispatch_time = copy.deepcopy(dispatch_time_copy)
            #                 process_time = copy.deepcopy(process_time_copy)
            #                 CPU_weight = copy.deepcopy(CPU_weight_copy)
            #                 total_latency = copy.deepcopy(total_latency_copy)
            #                 # Off_decision=copy.deepcopy(Off_decision_copy)
            #                 dispatch_decision = copy.deepcopy(
            #                     dispatch_decision_copy)
            #                 devices_dispatched = copy.deepcopy(
            #                     devices_dispatched_copy)
            #                 MDs_in_path_between_APs = copy.deepcopy(
            #                     MDs_in_path_between_APs_copy)

            #             # 如果本次更新过程并没有使AP m的策略改变，跳出更新循环
            #             #print("AP ",m,"更新后的分派情况:",devices_dispatched)
            #         if devices_dispatched == devices_dispatched_compare_copy:
            #             break
            #     # ____________________________________________________________________________________________________________________

            # '''if t_d >= MD_iteration_numbers-20:
            #     leiji_total_latency_wolf.append(sum(total_latency))
            #     #计算AP利用率，先算AP平均时延，再获取AP时延中的max，求两者之差
            #     this_simu_ave_max_delay_dif_wolf +=  np.var(AP_latency)#max(AP_latency) - sum(AP_latency)/M
            #     #
            #     this_ave_offload_MD_num_wolf += N - len(local_MD)
            #     #
            #     this_ave_APdelay_wolf += sum(AP_latency)
            #     #
            #     if t_d == MD_iteration_numbers-1:
            #         this_simu_ave_max_delay_dif_wolf = this_simu_ave_max_delay_dif_wolf/20
            #         ave_max_delay_dif_wolf += this_simu_ave_max_delay_dif_wolf
            #         #
            #         this_ave_offload_MD_num_wolf = this_ave_offload_MD_num_wolf/20
            #         ave_offload_MD_num_wolf += this_ave_offload_MD_num_wolf
            #         #
            #         this_ave_APdelay_wolf = this_ave_APdelay_wolf/20
            #         ave_APdelay_wolf += this_ave_APdelay_wolf'''

            # fig_total_delay.append(sum(total_latency))
            # ==========================  WoLF-PHC  =========更新策略和Q值=============================================================
            # print("(total_latency",total_latency)
            for i in range(N):
                # print(math.exp(device_Q_value[i][0]/lamda[i]))'
                # sum_Q_lamda[i]=0
                # sum_Q_lamda[i]=math.exp(device_Q_value[i][0]/lamda[i])
                sum_policy = 0.0
                # device_policy[i].clear()
                # device_policy[i].append(0)

                # 更新Q值
                if Off_decision[i] == i+M:
                    # device_Q_value[i][0]=(1-lamda[i])*device_Q_value[i][0]+lamda[i]*(1/total_latency[i])#(-total_latency[i])
                    device_Q_value[i][0] = (
                        (gameHistory[i][0]-1)*device_Q_value[i][0]+(1/total_latency[i]))/gameHistory[i][0]
                else:
                    # device_Q_value[i][Off_decision[i]+1]=(1-lamda[i])*device_Q_value[i][Off_decision[i]+1]+lamda[i]*(1/total_latency[i])#(-total_latency[i])
                    device_Q_value[i][Off_decision[i]+1] = ((gameHistory[i][Off_decision[i]+1]-1)*device_Q_value[i][Off_decision[i]+1]+(
                        1/total_latency[i]))/gameHistory[i][Off_decision[i]+1]

                # print(device_Q_value)

                # 计算平均策略期望值
                def func(x, y): return x*y
                result = map(func, ave_policy[i], device_Q_value[i])
                list_result_a = list(result)
                sum_ave_value[i] = sum(list_result_a)
                # 计算当前策略期望值
                resultc = map(func, device_policy[i], device_Q_value[i])
                list_result_c = list(resultc)
                sum_current_value[i] = sum(list_result_c)

                # 比较平均策略和当前策略
                if sum_ave_value[i] < sum_current_value[i]:
                    # 用小参数s_delta_win
                    s_delta[i] = s_delta_win
                else:
                    # 用大参数s_delta_lose
                    s_delta[i] = s_delta_lose

                # ---------先计算更新策略用的delta值------
                # 找到Q值最大的动作的索引，
                Q_rank = np.argsort(-np.array(device_Q_value[i]))
                # print('device_Q_value[',i,']',device_Q_value[i],'Q_rank',Q_rank)
                # 更新当前策略---------5-19---进行了修改（概率值归一化处理）
                sum_now = 0
                for num in range(M+1):
                    delta = 0
                    if len(devices[i][5]) == 0:
                        break
                    if num == 0 or num-1 in devices[i][5]:
                        if num == Q_rank[0]:
                            # print('选到Q大的动作了')
                            '''sum_delta=0
                            for a in range(M+1):
                                if a!=num and len(devices[i][5])!=0:
                                    sum_delta+=min(device_policy[i][a],s_delta[i]/( len(devices[i][5])+1 -1))
                            delta=sum_delta'''
                            delta = min(
                                device_policy[i][num], s_delta[i])  # /( len(devices[i][5])+1 -1))#sum_delta
                        else:
                            if len(devices[i][5]) != 0:
                                # /( len(devices[i][5])+1 -1))
                                delta = -min(device_policy[i][num], s_delta[i])
                    #print('delta of N=',i,'action=',num,':',delta)
                    device_policy[i][num] = device_policy[i][num]+delta
                # 概率归一化
                sum_thisCurrent = sum(device_policy[i])
                sum_change = 0
                for a in range(M+1):
                    if len(devices[i][5]) != 0:
                        if a != devices[i][5][-1]+1:
                            device_policy[i][a] = device_policy[i][a] / \
                                sum_thisCurrent
                            sum_change += device_policy[i][a]
                        else:
                            device_policy[i][a] = 1-sum_change
                    else:
                        break

            # if(t_d%50==0):
                # print("device_policy",device_policy)
                '''print('off_decision:',Off_decision)
                print("s_delta",s_delta)'''
                # print("total_latency",total_latency)

            # ------------------------------------判断策略是否收敛于一个阈值--------------------------------------
            '''if device_policy == MDs_policy_wolf:
                print_nip_wolf = False
                print("本单次实验中wolf算法已收敛，迭代次数为：",t_d+1) 
            else:'''
            for i in range(N):
                sum_wolf = 0
                for p in range(M+1):
                    sum_wolf += abs(device_policy[i][p]-MDs_policy_wolf[i][p])
                # print("sum_wolf:",sum_wolf)
                if sum_wolf <= epsino_wolf:
                    converge_wolf = True and converge_wolf
                else:
                    converge_wolf = False
                if converge_wolf == False:
                    break

            if converge_wolf == True:  # 202007282059-------------- and print_nip_wolf:  #表示已经输出过:
                print_nip_wolf = False
                if ave_round_wolf == ave_round:  # =================202007282046改=================================
                    print("本单次实验中wolf算法已收敛，迭代次数为：", t_d+1)
                    ave_iteration_wolf += t_d+1

            if print_nip_wolf == False:
                ave_round_wolf -= 1
                if ave_round_wolf > 0:
                    # =====================================================
                    ave_round_delay_wolf += sum(total_latency)
                    print("ave_round_delay_wolf", ave_round_delay_wolf)

                    # ------------------------20200815---------------------------------------
                    leiji_total_latency_wolf.append(sum(total_latency))
                    # 计算AP利用率，先算AP平均时延，再获取AP时延中的max，求两者之差
                    # max(AP_latency) - sum(AP_latency)/M
                    this_simu_ave_max_delay_dif_wolf += np.var(AP_latency)
                    #
                    this_ave_offload_MD_num_wolf += N - len(local_MD)
                    #
                    this_ave_APdelay_wolf += sum(AP_latency)
                    #
                    if ave_round_wolf == 1:
                        this_simu_ave_max_delay_dif_wolf = this_simu_ave_max_delay_dif_wolf/ave_round
                        ave_max_delay_dif_wolf += this_simu_ave_max_delay_dif_wolf
                        #
                        this_ave_offload_MD_num_wolf = this_ave_offload_MD_num_wolf/ave_round
                        ave_offload_MD_num_wolf += this_ave_offload_MD_num_wolf
                        #
                        this_ave_APdelay_wolf = this_ave_APdelay_wolf/ave_round
                        ave_APdelay_wolf += this_ave_APdelay_wolf

        else:
            fig_total_delay.append(sum(total_latency))

        # -------------------------------------------------------------------------------------------------------
        # ======================================下面的if是我自己的旧方法=============================================================
        if converge_old and ave_round_old > 0 and notEndTime:  # and print_nip_old :
            for i in range(N):
                # 记录MD2的策略
                '''if simualtion_index==0:
                    device_old_y1[i].append(device_policy_old[i][0])
                    device_old_y2[i].append(device_policy_old[i][1])
                    device_old_y3[i].append(device_policy_old[i][2])
                    device_old_y4[i].append(device_policy_old[i][3])
                    device_old_y5[i].append(device_policy_old[i][4])
                    device_old_y6[i].append(device_policy_old[i][5])'''

                dispatch_time_old[i] = 0
                process_time_old[i] = 0

                for m in range(M):
                    trans_time_old[i][m] = 0
                local_MD_old.append(i+M)
                # device根据策略选动作#,p=device_policy[i].ravel()
                Off_decision_old[i] = np.random.choice(
                    off_decision_index[i], p=device_policy_old[i])

                # print(i+M,"offloading_decision:",Off_decision[i])
                if Off_decision_old[i] < M:
                    dispatch_decision_old[i] = Off_decision_old[i]
                    local_MD_old.remove(i+M)
                    devices_in_AP_old[Off_decision_old[i]].append(
                        i+M)  # 选择卸载到某AP的设备号
                    tasks_input_in_AP_old[Off_decision_old[i]].append(
                        tasks[i][1])  # 选择卸载到某AP的任务inputsize
                    tasks_cpu_in_AP_old[Off_decision_old[i]].append(
                        tasks[i][2])  # 选择卸载到某AP的任务cpucycles
            # print("devices_in_AP_old",devices_in_AP_old)
            # print('off_decision_old:',Off_decision_old)
            for m in range(M):
                for n in range(M):
                    devices_dispatched_old[m][n].clear()
                devices_dispatched_old[m][m] = copy.deepcopy(
                    devices_in_AP_old[m])
            trans_dispatch_process_delay.s_delay(
                N, M, devices, APs, tasks, devices_in_AP_old, trans_rate, trans_time_old, wireless_weight_old, actual_rate_old)
            trans_dispatch_process_delay.c_delay(
                M, N, APs, devices, tasks, devices_dispatched_old, process_time_old, CPU_weight_old)
            for i in local_MD_old:  # 本地执行的设备的时延存储
                total_latency_old[i-M] = process_time_old[i-M]
            # ------------------------------------------------------------------------------------------
            # ————————————————————————2020-7-17—————----   Old algorithm ----—更新每个电缆上的MD—————————————————————————————————
            MDs_in_path_between_APs_old = [[[]
                                            for i in range(M)]for j in range(M)]
            for m in range(M):
                for n in range(M):
                    for i in devices_dispatched_old[m][n]:
                        for m_ in range(len(path_between_APs[m][n])-1):
                            if m_ != len(path_between_APs[m][n])-1:
                                MDs_in_path_between_APs_old[path_between_APs[m]
                                                            [n][m_]][path_between_APs[m][n][m_+1]].append(i)

            # ——————————————————————————————————————————————————————————————————————
            trans_dispatch_process_delay.d_delay(N, M, tasks, devices_dispatched_old, dispatch_time_old,
                                                 dispatch_road_width, wired_weight_old, path_between_APs, MDs_in_path_between_APs_old)

            # ------------------------------------------------------------------------------------------
            for m in range(M):  # 初始dispatch决定带来的时延
                for i in devices_in_AP_old[m]:
                    total_latency_old[i-M] = trans_time_old[i-M][m] + \
                        dispatch_time_old[i-M]+process_time_old[i-M]
                    AP_latency_old[m] = AP_latency_old[m] + \
                        total_latency_old[i-M]
            # print("AP初始收到的任务:",devices_dispatched_old)
            # ----------------------------------------------------------------------------------------
            '''if t_d>=MD_iteration_numbers-20:
                leiji_total_latency_old.append(sum(total_latency_old))
                #计算AP利用率，先算AP平均时延，再获取AP时延中的max，求两者之差
                this_simu_ave_max_delay_dif_old +=  np.var(AP_latency_old)#max(AP_latency_old) - sum(AP_latency_old)/M
                #
                this_ave_offload_MD_num_old += N - len(local_MD_old)
                #
                this_ave_APdelay_old += sum(AP_latency_old)
                #
                if t_d == MD_iteration_numbers-1:
                    this_simu_ave_max_delay_dif_old  = this_simu_ave_max_delay_dif_old/20
                    ave_max_delay_dif_old  += this_simu_ave_max_delay_dif_old
                    #
                    this_ave_offload_MD_num_old = this_ave_offload_MD_num_old/20
                    ave_offload_MD_num_old += this_ave_offload_MD_num_old
                    #
                    this_ave_APdelay_old = this_ave_APdelay_old/20
                    ave_APdelay_old += this_ave_APdelay_old'''
            # # -----------------------------AP迭代------------------------------------------------------------------------------------------
            # # 最开始的初始化dispatch decisions就是AP自己执行自己的，然后如果有AP能通过分派任务使得自身的任务时延减小，令这些AP中 \sum(CPUcyccles)/\sum(inputsiza)最大的AP先更新策略,比例要不要加期望要再想一下
            # # \sum(CPUcyccles)/\sum(inputsiza)的存储列表
            # tasks_input_cpu_ratio_rank = [[]for i in range(M)]

            # # ============================开始AP的决策更新=(把对比方法的也同时算出来)=====================================
            # for m in range(M):  # 计算并降序排序\sum(CPUcyccles)/\sum(inputsiza)？？？？？？论文里分母是AP自己的CPU频率值
            #     if sum(tasks_input_in_AP_old[m]) > 0:
            #         tasks_input_cpu_ratio_rank[m] = (
            #             sum(tasks_cpu_in_AP_old[m])/APs[m][4])  # sum(tasks_input_in_AP[m])
            #     else:
            #         tasks_input_cpu_ratio_rank[m] = 0
            #     # tasks_input_cpu_ratio_rank_copy.append(sum(tasks_cpu_in_AP[m])/sum(tasks_input_in_AP[m]))

            #     # ---------------------------------------------------------------------------------------------
            # # print(tasks_input_cpu_ratio_rank)
            # tasks_input_cpu_ratio_rank_copy = np.argsort(
            #     -np.array(tasks_input_cpu_ratio_rank))
            # #
            # # print(tasks_input_cpu_ratio_rank_copy)
            # # print("tasks_cpu_in_AP",tasks_cpu_in_AP)
            # for t_A in range(1):
            #     # print("t_A",t_A)
            #     while(True):  # 只要AP策略没收敛，就一直更新
            #         devices_dispatched_compare_copy = copy.deepcopy(
            #             devices_dispatched_old)
            #         # print("devices_dispatched_compare_copy:",devices_dispatched_compare_copy)
            #         for h in range(M):  # 按排序使AP异步更新
            #             # m为\sum(CPUcyccles)/\sum(inputsiza)第h大的AP
            #             m = tasks_input_cpu_ratio_rank_copy[h]
            #             if tasks_input_cpu_ratio_rank[m] == 0:
            #                 break
            #             # task_cpu_in_AP_copy是降序排序的原列表从大到小的索引
            #             task_cpu_in_AP_copy = np.argsort(
            #                 -np.array(tasks_cpu_in_AP_old[m]))
            #             # print(task_cpu_in_AP_copy)

            #             # devices_dispatched_compare_copy=copy.deepcopy(devices_dispatched)
            #             #print("devices_dispatched_compare_copy in h=",h,":",devices_dispatched_compare_copy)
            #             for j in range(len(set(devices_in_AP_old[m]))):
            #                 # print(task_cpu_in_AP_copy[j])
            #                 max_i_cpu = task_cpu_in_AP_copy[j]
            #                 # print(max_i_cpu)
            #                 # 求出m上cpu cycles最大的任务所属设备号
            #                 max_i = devices_in_AP_old[m][max_i_cpu]
            #                 # print(max_i)
            #                 # latency_maxi_copy=copy.deepcopy(total_latency[max_i-M]) #把max_i原始时延先拷贝下来
            #                 # 下面这一顿deepcopy是为了存储到最好的分派决策之后再更新原始的策略-----
            #                 dispatch_time_copy = copy.deepcopy(
            #                     dispatch_time_old)
            #                 process_time_copy = copy.deepcopy(process_time_old)
            #                 CPU_weight_copy = copy.deepcopy(CPU_weight_old)
            #                 total_latency_copy = copy.deepcopy(
            #                     total_latency_old)
            #                 AP_latency_copy = copy.deepcopy(AP_latency_old)
            #                 # Off_decision_copy=copy.deepcopy(Off_decision)
            #                 dispatch_decision_copy = copy.deepcopy(
            #                     dispatch_decision_old)
            #                 devices_dispatched_copy = copy.deepcopy(
            #                     devices_dispatched_old)  # -------------
            #                 MDs_in_path_between_APs_copy = copy.deepcopy(
            #                     MDs_in_path_between_APs_old)

            #                 for n in range(M):  # 循环，为max_i挑出最好的修改分派方案 and n!=m
            #                     # 下面这一顿deepcopy是使得每次计算针对当前max_i的分派决定需要有好效果才更新到原有的copy版本-----
            #                     dispatch_time_copy2 = copy.deepcopy(
            #                         dispatch_time_copy)
            #                     process_time_copy2 = copy.deepcopy(
            #                         process_time_copy)
            #                     CPU_weight_copy2 = copy.deepcopy(
            #                         CPU_weight_copy)
            #                     total_latency_copy2 = copy.deepcopy(
            #                         total_latency_copy)
            #                     AP_latency_copy2 = copy.deepcopy(
            #                         AP_latency_copy)
            #                     # Off_decision_copy2=copy.deepcopy(Off_decision_copy)
            #                     dispatch_decision_copy2 = copy.deepcopy(
            #                         dispatch_decision_copy)
            #                     devices_dispatched_copy2 = copy.deepcopy(
            #                         devices_dispatched_copy)  # -------------
            #                     MDs_in_path_between_APs_copy2 = copy.deepcopy(
            #                         MDs_in_path_between_APs_copy)

            #                     # 把cpucycles最大的设备先从原分配方案中删去，然后换新的计算延迟
            #                     devices_dispatched_copy2[m][dispatch_decision_copy2[max_i-M]].remove(
            #                         max_i)
            #                     # Off_decision_copy2[max_i-M]=n
            #                     # ————————————————————————2020-7-17——————虚拟移除原来分派选路上的MD——————————
            #                     for m_ in range(len(path_between_APs[m][dispatch_decision_copy2[max_i-M]])-1):
            #                         if m_ != len(path_between_APs[m][dispatch_decision_copy2[max_i-M]])-1:
            #                             MDs_in_path_between_APs_copy2[path_between_APs[m][dispatch_decision_copy2[max_i-M]]
            #                                                           [m_]][path_between_APs[m][dispatch_decision_copy2[max_i-M]][m_+1]].remove(max_i)
            #                     # ——————————————————————————————————————————————————————————————————————
            #                     dispatch_decision_copy2[max_i-M] = n
            #                     # 接下来计算把max_i放到n上产生的时延
            #                     devices_dispatched_copy2[m][dispatch_decision_copy2[max_i-M]].append(
            #                         max_i)
            #                     # trans_dispatch_process_delay.d_delay(max_i,M,tasks,devices_dispatched_copy2,dispatch_time_copy2,dispatch_rate)
            #                     # ————————————————————————2020-7-17——————虚拟更新当前选择分派选路上的MD——————————
            #                     for m_ in range(len(path_between_APs[m][n])-1):
            #                         MDs_in_path_between_APs_copy2[path_between_APs[m][dispatch_decision_copy2[max_i-M]]
            #                                                       [m_]][path_between_APs[m][dispatch_decision_copy2[max_i-M]][m_+1]].append(max_i)
            #                     # ——————————————————————————————————————————————————————————————————————
            #                     trans_dispatch_process_delay.d_delay(
            #                         N, M, tasks, devices_dispatched_copy2, dispatch_time_old, dispatch_road_width, wired_weight_old, path_between_APs, MDs_in_path_between_APs_copy2)
            #                     trans_dispatch_process_delay.c_delay(
            #                         M, N, APs, devices, tasks, devices_dispatched_copy2, process_time_copy2, CPU_weight_copy2)
            #                     AP_latency_try = 0
            #                     devices_not_in = []
            #                     for i in range(N):
            #                         devices_not_in.append(i+M)
            #                     for i in devices_in_AP_old[m]:
            #                         devices_not_in.remove(i)
            #                         #print("----------before copy----------")
            #                         total_latency_copy2[i-M] = trans_time_old[i-M][m] + \
            #                             dispatch_time_copy2[i-M] + \
            #                             process_time_copy2[i-M]
            #                         #print("----------after copy----------")
            #                         AP_latency_try = AP_latency_try + \
            #                             total_latency_copy2[i-M]
            #                     for i in devices_not_in:
            #                         total_latency_copy2[i-M] = sum(
            #                             trans_time_old[i-M])+dispatch_time_copy2[i-M]+process_time_copy2[i-M]
            #                     # 如果这个选择是比原来的好，那就更新策略，保存时延结果
            #                     if AP_latency_try < AP_latency_copy2[m]:
            #                         #print("update in n")
            #                         #print("==========before copy============")
            #                         AP_latency_copy[m] = copy.deepcopy(
            #                             AP_latency_try)
            #                         dispatch_time_copy = copy.deepcopy(
            #                             dispatch_time_copy2)
            #                         process_time_copy = copy.deepcopy(
            #                             process_time_copy2)
            #                         CPU_weight_copy = copy.deepcopy(
            #                             CPU_weight_copy2)
            #                         total_latency_copy = copy.deepcopy(
            #                             total_latency_copy2)
            #                         # Off_decision_copy=copy.deepcopy(Off_decision_copy2)
            #                         dispatch_decision_copy = copy.deepcopy(
            #                             dispatch_decision_copy2)
            #                         devices_dispatched_copy = copy.deepcopy(
            #                             devices_dispatched_copy2)
            #                         MDs_in_path_between_APs_copy = copy.deepcopy(
            #                             MDs_in_path_between_APs_copy2)
            #                         #print("==========after copy============")
            #                 # 下面的deepcopy把针对与当前max_i的最好结果进行更新
            #                 AP_latency_old[m] = copy.deepcopy(
            #                     AP_latency_copy[m])
            #                 dispatch_time_old = copy.deepcopy(
            #                     dispatch_time_copy)
            #                 process_time_old = copy.deepcopy(process_time_copy)
            #                 CPU_weight_old = copy.deepcopy(CPU_weight_copy)
            #                 total_latency_old = copy.deepcopy(
            #                     total_latency_copy)
            #                 # Off_decision=copy.deepcopy(Off_decision_copy)
            #                 dispatch_decision_old = copy.deepcopy(
            #                     dispatch_decision_copy)
            #                 devices_dispatched_old = copy.deepcopy(
            #                     devices_dispatched_copy)
            #                 MDs_in_path_between_APs_old = copy.deepcopy(
            #                     MDs_in_path_between_APs_copy)

            #         if devices_dispatched_old == devices_dispatched_compare_copy:
            #             break

            # fig_total_delay_old.append(sum(total_latency_old))
            # ------------------------------------------------------------------------------------
            '''if t_d>=MD_iteration_numbers-20:
                leiji_total_latency_old.append(sum(total_latency_old))
                #计算AP利用率，先算AP平均时延，再获取AP时延中的max，求两者之差
                this_simu_ave_max_delay_dif_old +=  np.var(AP_latency_old)#max(AP_latency_old) - sum(AP_latency_old)/M
                #
                this_ave_offload_MD_num_old += N - len(local_MD_old)
                #
                this_ave_APdelay_old += sum(AP_latency_old)
                #
                if t_d == MD_iteration_numbers-1:
                    this_simu_ave_max_delay_dif_old  = this_simu_ave_max_delay_dif_old/20
                    ave_max_delay_dif_old  += this_simu_ave_max_delay_dif_old
                    #
                    this_ave_offload_MD_num_old = this_ave_offload_MD_num_old/20
                    ave_offload_MD_num_old += this_ave_offload_MD_num_old
                    #
                    this_ave_APdelay_old = this_ave_APdelay_old/20
                    ave_APdelay_old += this_ave_APdelay_old'''

            # ============================================旧方法更新策略和Q值===================================================================
            # print("(total_latency",total_latency)
            for i in range(N):
                # print(math.exp(device_Q_value[i][0]/lamda[i]))'
                sum_Q_lamda[i] = 0
                sum_Q_lamda[i] = math.exp(device_Q_value[i][0]/lamda[i])
                sum_policy = 0.0
                device_policy_old[i].clear()
                device_policy_old[i].append(0)
                for m in range(M):
                    # print(math.exp(device_Q_value[i][m+1]/lamda[i]))
                    sum_Q_lamda[i] = sum_Q_lamda[i] + \
                        math.exp(device_Q_value_old[i][m+1]/lamda[i])
                # print("sum_Q_lamda[",i+M,"]",sum_Q_lamda[i])
                for m in range(M):
                    if m in devices[i][5]:
                        x = (
                            math.exp(device_Q_value_old[i][m+1]/lamda[i]))/sum_Q_lamda[i]
                        device_policy_old[i].append(x)
                        sum_policy = sum_policy+x
                        if Off_decision_old[i] == m:
                            device_Q_value_old[i][m+1] = (
                                1-theda[i])*device_Q_value_old[i][m+1]+theda[i]*(1/total_latency_old[i])
                    else:
                        device_policy_old[i].append(0)
                    #print("sum_policy of",i+M,":",sum_policy)
                # (math.exp(device_Q_value[i][0]/lamda[i]))/sum_Q_lamda[i]
                device_policy_old[i][0] = 1-sum_policy
                if Off_decision_old[i] == i+M:
                    device_Q_value_old[i][0] = (
                        1-theda[i])*device_Q_value_old[i][0]+theda[i]*(1/total_latency_old[i])
            # ===========================================================================
            # ------------------------------------判断策略是否收敛于一个阈值--------------------------------------
            '''if device_policy_old == MDs_policy_old:
                print_nip_old = False
                print("本单次实验中old算法已收敛，迭代次数为：",t_d+1) 
            else:'''
            for i in range(N):
                sum_old = 0
                for p in range(M+1):
                    sum_old += abs(device_policy_old[i]
                                   [p]-MDs_policy_old[i][p])
                # print("sum_old",sum_old)
                if sum_old <= epsino_old:
                    converge_old = True and converge_old
                else:
                    converge_old = False
                if converge_old == False:
                    break

            if converge_old == True:  # and print_nip_old:
                print_nip_old = False  # 表示已经收敛
                if ave_round_old == ave_round:
                    print("本单次实验中Q-based算法已收敛，迭代次数为：", t_d+1)
                    ave_iteration_old += t_d+1

            if print_nip_old == False:
                ave_round_old -= 1
                if ave_round_old > 0:
                    ave_round_delay_old += sum(total_latency_old)
                    print("ave_round_delay_old", ave_round_delay_old)

                    leiji_total_latency_old.append(sum(total_latency_old))
                    # 计算AP利用率，先算AP平均时延，再获取AP时延中的max，求两者之差
                    # max(AP_latency_old) - sum(AP_latency_old)/M
                    this_simu_ave_max_delay_dif_old += np.var(AP_latency_old)
                    #
                    this_ave_offload_MD_num_old += N - len(local_MD_old)
                    #
                    this_ave_APdelay_old += sum(AP_latency_old)
                    #
                    if ave_round_old == 1:
                        this_simu_ave_max_delay_dif_old = this_simu_ave_max_delay_dif_old/ave_round
                        ave_max_delay_dif_old += this_simu_ave_max_delay_dif_old
                        #
                        this_ave_offload_MD_num_old = this_ave_offload_MD_num_old/ave_round
                        ave_offload_MD_num_old += this_ave_offload_MD_num_old
                        #
                        this_ave_APdelay_old = this_ave_APdelay_old/ave_round
                        ave_APdelay_old += this_ave_APdelay_old

        else:
            fig_total_delay_old.append(sum(total_latency_old))

        if ave_round_old == 0 and ave_round_wolf == 0:
            notEndTime = False
            # break
        converge_old = True
        converge_wolf = True

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Greedy  offloading!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for i in range(N):
            dispatch_time_GO[i] = 0
            process_time_GO[i] = 0
            for m in range(M):
                trans_time_GO[i][m] = 0
            local_MD_GO.append(i+M)
            # ！！！！！！！！！！！！！！在Greedy offloading 中的卸载决定(max -Q until 本身算法收敛 + AP 按BR-分派)！！！！！！！！！！！！！
            device_Q_value_GO_copy = np.array(device_Q_value_GO[i])
            d_Q_GO_copy2 = np.argsort(-device_Q_value_GO_copy)
            # print("d_Q_GO_copy2",d_Q_GO_copy2)
            # print("d_Q_GO_copy2[0]",d_Q_GO_copy2[0])
            if t_d == 0:
                # print(len(set(off_real_index[i])))
                index_OFF_GO = random.randint(0, len(set(off_real_index[i]))-1)
                # print("index_OFF_GO",index_OFF_GO)
                Off_decision_GO[i] = off_real_index[i][index_OFF_GO]
            else:
                Off_decision_GO[i] = off_decision_index[i][d_Q_GO_copy2[0]]
            # print("Off_decision_GO[",i+M,"]",Off_decision_GO[i])
            if Off_decision_GO[i] < M:
                dispatch_decision_GO[i] = Off_decision_GO[i]
                local_MD_GO.remove(i+M)
                devices_in_AP_GO[Off_decision_GO[i]].append(
                    i+M)  # 选择卸载到某AP的设备号
                tasks_input_in_AP_GO[Off_decision_GO[i]].append(
                    tasks[i][1])  # 选择卸载到某AP的任务inputsize
                tasks_cpu_in_AP_GO[Off_decision_GO[i]].append(
                    tasks[i][2])  # 选择卸载到某AP的任务cpucycles
        for m in range(M):
            for n in range(M):
                devices_dispatched_GO[m][n].clear()
            devices_dispatched_GO[m][m] = copy.deepcopy(devices_in_AP_GO[m])
        trans_dispatch_process_delay.s_delay(
            N, M, devices, APs, tasks, devices_in_AP_GO, trans_rate, trans_time_GO, wireless_weight_GO, actual_rate_GO)
        trans_dispatch_process_delay.c_delay(
            M, N, APs, devices, tasks, devices_dispatched_GO, process_time_GO, CPU_weight_GO)

        # ————————————————————————2020-7-17—————----   Greedy algorithm ----—更新每个电缆上的MD————————————————————————————————
        # print("devices_dispatched_GO",devices_dispatched_GO)
        MDs_in_path_between_APs_GO = [[[]for i in range(M)]for j in range(M)]
        for m in range(M):
            for n in range(M):
                for i in devices_dispatched_GO[m][n]:
                    for m_ in range(len(path_between_APs[m][n])-1):
                        if m_ != len(path_between_APs[m][n])-1:
                            MDs_in_path_between_APs_GO[path_between_APs[m][n]
                                                       [m_]][path_between_APs[m][n][m_+1]].append(i)
        # print('--',MDs_in_path_between_APs_GO)
        # ——————————————————————————————————————————————————————————————————————
        trans_dispatch_process_delay.d_delay(N, M, tasks, devices_dispatched_GO, dispatch_time,
                                             dispatch_road_width, wired_weight_GO, path_between_APs, MDs_in_path_between_APs_GO)
        # print("local_MD_GO",local_MD_GO)
        for j in local_MD_GO:  # 本地执行的设备的时延存储
            total_latency_GO[j-M] = process_time_GO[j-M]
            # print("total_latency_GO[",j,"]:",total_latency_GO[j-M])
        for m in range(M):  # 初始dispatch决定带来的时延
            for j in devices_in_AP_GO[m]:
                # trans_dispatch_process_delay.d_delay(i,M,tasks,devices_dispatched_GO,dispatch_time_GO,dispatch_rate)
                total_latency_GO[j-M] = trans_time_GO[j-M][m] + \
                    dispatch_time_GO[j-M]+process_time_GO[j-M]
                AP_latency_GO[m] = AP_latency_GO[m]+total_latency_GO[j-M]
        # print("original_process_latency_GO:",process_time_GO)
        # print("original_total_latency_GO:",total_latency_GO)
        leiji_total_latency_GO.append(sum(total_latency_GO))
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Fast  offloading!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if t_d == 0:
            for i in range(N):
                dispatch_time_FO[i] = 0
                process_time_FO[i] = 0
                for m in range(M):
                    trans_time_FO[i][m] = 0
                local_MD_FO.append(i+M)

                # ！！！！！！！！！！！！！！在Fast offloading 中的卸载决定(MD取原始传输速率最大的卸载 + AP分派)！！！！！！！！！！！！！
                trans_ratesort = np.array(trans_rate[i])
                trans_ratesort_copy = np.argsort(-trans_ratesort)
                if sum(trans_rate[i]) > 0:
                    Off_decision_FO[i] = trans_ratesort_copy[0]
                else:
                    Off_decision_FO[i] = i+M

                if Off_decision_FO[i] < M:
                    dispatch_decision_FO[i] = Off_decision_FO[i]
                    local_MD_FO.remove(i+M)
                    devices_in_AP_FO[Off_decision_FO[i]].append(
                        i+M)  # 选择卸载到某AP的设备号
                    tasks_input_in_AP_FO[Off_decision_FO[i]].append(
                        tasks[i][1])  # 选择卸载到某AP的任务inputsize
                    tasks_cpu_in_AP_FO[Off_decision_FO[i]].append(
                        tasks[i][2])  # 选择卸载到某AP的任务cpucycles

            for m in range(M):
                for n in range(M):
                    devices_dispatched_FO[m][n].clear()
                devices_dispatched_FO[m][m] = copy.deepcopy(
                    devices_in_AP_FO[m])
            trans_dispatch_process_delay.s_delay(
                N, M, devices, APs, tasks, devices_in_AP_FO, trans_rate, trans_time_FO, wireless_weight_FO, actual_rate_FO)
            trans_dispatch_process_delay.c_delay(
                M, N, APs, devices, tasks, devices_dispatched_FO, process_time_FO, CPU_weight_FO)
            # ————————————————————————2020-7-17—————----   Fast algorithm ----—更新每个电缆上的MD—————————————————————————————————
            MDs_in_path_between_APs_FO = [[[]
                                           for i in range(M)]for j in range(M)]
            for m in range(M):
                for n in range(M):
                    for i in devices_dispatched_FO[m][n]:
                        for m_ in range(len(path_between_APs[m][n])-1):
                            if m_ != len(path_between_APs[m][n])-1:
                                MDs_in_path_between_APs_FO[path_between_APs[m][n]
                                                           [m_]][path_between_APs[m][n][m_+1]].append(i)
            # ——————————————————————————————————————————————————————————————————————
            trans_dispatch_process_delay.d_delay(N, M, tasks, devices_dispatched_FO, dispatch_time,
                                                 dispatch_road_width, wired_weight_FO, path_between_APs, MDs_in_path_between_APs_FO)
            for j in local_MD_FO:  # 本地执行的设备的时延存储
                total_latency_FO[j-M] = process_time_FO[j-M]
            for m in range(M):  # 初始dispatch决定带来的时延
                for j in devices_in_AP_FO[m]:
                    # trans_dispatch_process_delay.d_delay(j,M,tasks,devices_dispatched_FO,dispatch_time_FO,dispatch_rate)
                    total_latency_FO[j-M] = trans_time_FO[j-M][m] + \
                        dispatch_time_FO[j-M]+process_time_FO[j-M]
                    AP_latency_FO[m] = AP_latency_FO[m]+total_latency_FO[j-M]
        leiji_total_latency_FO.append(sum(total_latency_FO))
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Directed  offloading!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # print('AP_latency_DO',AP_latency_DO)
        if converge_DO and ave_round_DO > 0 and notEndTime_DO:  # and print_nip_wolf:202027282043改
            if t_d == 0:
                Off_decision_DO = copy.deepcopy(Off_decision)
                for i in range(N):
                    dispatch_time_DO[i] = 0
                    process_time_DO[i] = 0
                    for m in range(M):
                        trans_time_DO[i][m] = 0
                    local_MD_DO.append(i+M)

                    if Off_decision_DO[i] < M:
                        dispatch_decision_DO[i] = Off_decision_DO[i]
                        local_MD_DO.remove(i+M)
                        devices_in_AP_DO[Off_decision_DO[i]].append(
                            i+M)  # 选择卸载到某AP的设备号
                        tasks_input_in_AP_DO[Off_decision_DO[i]].append(
                            tasks[i][1])  # 选择卸载到某AP的任务inputsize
                        tasks_cpu_in_AP_DO[Off_decision_DO[i]].append(
                            tasks[i][2])  # 选择卸载到某AP的任务cpucycles
                    for m in range(M):
                        for n in range(M):
                            devices_dispatched_DO[m][n].clear()
                        devices_dispatched_DO[m][m] = copy.deepcopy(
                            devices_in_AP_DO[m])
            else:
                for i in range(N):
                    dispatch_time_DO[i] = 0
                    process_time_DO[i] = 0
                    for m in range(M):
                        trans_time_DO[i][m] = 0
                    local_MD_DO.append(i+M)
                    # ！！！！！！！！！！！！！！在Direction offloading 中的卸载决定(自己的卸载方法 + AP不分派)！！！！！！！！！！！！！
                    Off_decision_DO[i] = np.random.choice(
                        off_decision_index[i], p=device_policy_DO[i])

                    if Off_decision_DO[i] < M:
                        dispatch_decision_DO[i] = Off_decision_DO[i]
                        local_MD_DO.remove(i+M)
                        devices_in_AP_DO[Off_decision_DO[i]].append(
                            i+M)  # 选择卸载到某AP的设备号
                        tasks_input_in_AP_DO[Off_decision_DO[i]].append(
                            tasks[i][1])  # 选择卸载到某AP的任务inputsize
                        tasks_cpu_in_AP_DO[Off_decision_DO[i]].append(
                            tasks[i][2])  # 选择卸载到某AP的任务cpucycles
                for m in range(M):
                    for n in range(M):
                        devices_dispatched_DO[m][n].clear()
                    devices_dispatched_DO[m][m] = copy.deepcopy(
                        devices_in_AP_DO[m])
            # print("t_d",t_d,"-devices_in_AP_DO",devices_in_AP_DO)
            trans_dispatch_process_delay.s_delay(
                N, M, devices, APs, tasks, devices_in_AP_DO, trans_rate, trans_time_DO, wireless_weight_DO, actual_rate_DO)
            trans_dispatch_process_delay.c_delay(
                M, N, APs, devices, tasks, devices_dispatched_DO, process_time_DO, CPU_weight_DO)
            # trans_dispatch_process_delay.d_delay(N,M,tasks,devices_dispatched_DO,dispatch_time,dispatch_road_width,wired_weight_DO)
            for i in local_MD_DO:  # 本地执行的设备的时延存储
                total_latency_DO[i-M] = process_time_DO[i-M]
            for m in range(M):  # 初始dispatch决定带来的时延
                AP_latency_DO[m] = 0
                for i in devices_in_AP_DO[m]:
                    # trans_dispatch_process_delay.d_delay(i,M,tasks,devices_dispatched_DO,dispatch_time_DO,dispatch_rate)
                    # +dispatch_time_DO[i-M]+process_time_DO[i-M]
                    total_latency_DO[i-M] = trans_time_DO[i -
                                                          M][m]+process_time_DO[i-M]
                    AP_latency_DO[m] = AP_latency_DO[m]+total_latency_DO[i-M]
            # if t_d>=MD_iteration_numbers-30:
                # leiji_total_latency_DO.append(sum(total_latency_DO))
            leiji_total_latency_DO.append(sum(total_latency_DO))

            # ------------------------------------------------DO---策略更新-------------------------------------------
            for i in range(N):
                # print(math.exp(device_Q_value[i][0]/lamda[i]))
                sum_Q_lamda_DO[i] = 0
                sum_Q_lamda_DO[i] = math.exp(device_Q_value_DO[i][0]/lamda[i])
                sum_policy_DO = 0.0
                device_policy_DO[i].clear()
                device_policy_DO[i].append(0)
                for m in range(M):
                    sum_Q_lamda_DO[i] = sum_Q_lamda_DO[i] + \
                        math.exp(device_Q_value_DO[i][m+1]/lamda[i])
                # print(sum_Q_lamda_DO[i])
                # print("sum_Q_lamda_DO[",i+M,"]",sum_Q_lamda_DO[i])
                for m in range(M):
                    if m in devices[i][5]:
                        # print(math.exp(device_Q_value[i][m+1]/lamda[i]))
                        x = (
                            math.exp(device_Q_value_DO[i][m+1]/lamda[i]))/sum_Q_lamda_DO[i]
                        # device_policy_DO[i].append(x)
                        if (sum_policy_DO+x) > 1:
                            device_policy_DO[i].append(1-sum_policy_DO)
                            sum_policy_DO = 1
                        else:
                            device_policy_DO[i].append(x)
                            sum_policy_DO = sum_policy_DO+x
                        # sum_policy_DO=sum_policy_DO+x
                        if Off_decision_DO[i] == m:
                            device_Q_value_DO[i][m+1] = (
                                1-theda[i])*device_Q_value_DO[i][m+1]+theda[i]*(1/total_latency_DO[i])
                    else:
                        device_policy_DO[i].append(0)
                    #print("sum_policy of",i+M,":",sum_policy)
                # (math.exp(device_Q_value[i][0]/lamda[i]))/sum_Q_lamda[i]
                device_policy_DO[i][0] = 1-sum_policy_DO
                if Off_decision_DO[i] == i+M:
                    device_Q_value_DO[i][0] = (
                        1-theda[i])*device_Q_value_DO[i][0]+theda[i]*(1/total_latency_DO[i])
            # print("t_d",t_d,"-device_policy_DO:",device_policy_DO)
            # ----------------------------------------------------------------------------------
            # ------------------------------------判断策略是否收敛于一个阈值--------------------------------------
            '''if device_policy_old == MDs_policy_old:
                print_nip_old = False
                print("本单次实验中old算法已收敛，迭代次数为：",t_d+1) 
            else:'''
            for i in range(N):
                sum_DO = 0
                for p in range(M+1):
                    sum_DO += abs(device_policy_DO[i][p]-MDs_policy_DO[i][p])
                # print("sum_old",sum_old)
                if sum_DO <= epsino_DO:
                    converge_DO = True and converge_DO
                else:
                    converge_DO = False
                if converge_DO == False:
                    break

            if converge_DO == True:  # and print_nip_old:
                print_nip_DO = False  # 表示已经收敛
                if ave_round_DO == ave_round:
                    print("本单次实验中DO算法已收敛，迭代次数为：", t_d+1)
                    ave_iteration_DO += t_d+1

            if print_nip_DO == False:
                ave_round_DO -= 1
                if ave_round_DO > 0:
                    ave_round_delay_DO += sum(total_latency_DO)
                    # print("ave_round_delay_DO",ave_round_delay_DO)

                    # ------------------------20200815---------------------------------------
                    leiji_total_latency_DO.append(sum(total_latency_DO))
                    # 计算AP利用率，先算AP平均时延，再获取AP时延中的max，求两者之差
                    # max(AP_latency_DO) - sum(AP_latency_DO)/M
                    this_simu_ave_max_delay_dif_DO += np.var(AP_latency_DO)
                    #
                    this_ave_offload_MD_num_DO += N - len(local_MD_DO)
                    #
                    this_ave_APdelay_DO += sum(AP_latency_DO)
                    #
                    if ave_round_DO == 1:
                        this_simu_ave_max_delay_dif_DO = this_simu_ave_max_delay_dif_DO/ave_round
                        ave_max_delay_dif_DO += this_simu_ave_max_delay_dif_DO
                        #
                        this_ave_offload_MD_num_DO = this_ave_offload_MD_num_DO/ave_round
                        ave_offload_MD_num_DO += this_ave_offload_MD_num_DO
                        #
                        this_ave_APdelay_DO = this_ave_APdelay_DO/ave_round
                        ave_APdelay_DO += this_ave_APdelay_DO

        else:
            fig_total_delay_DO.append(sum(total_latency_DO))

        if ave_round_DO == 0:
            notEndTime_DO = False
            # break
        converge_DO = True

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Random offloading algorithm !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # print('AP_latency_RO',AP_latency_RO)
        if t_d == 0:
            Off_decision_RO = copy.deepcopy(Off_decision)

            for i in range(N):
                dispatch_time_RO[i] = 0
                process_time_RO[i] = 0
                for m in range(M):
                    trans_time_RO[i][m] = 0
                local_MD_RO.append(i+M)
                if Off_decision_RO[i] < M:
                    dispatch_decision_RO[i] = Off_decision_RO[i]
                    local_MD_RO.remove(i+M)
                    devices_in_AP_RO[Off_decision_RO[i]].append(
                        i+M)  # 选择卸载到某AP的设备号
                    tasks_input_in_AP_RO[Off_decision_RO[i]].append(
                        tasks[i][1])  # 选择卸载到某AP的任务inputsize
                    tasks_cpu_in_AP_RO[Off_decision_RO[i]].append(
                        tasks[i][2])  # 选择卸载到某AP的任务cpucycles
            for m in range(M):
                for n in range(M):
                    devices_dispatched_RO[m][n].clear()
                devices_dispatched_RO[m][m] = copy.deepcopy(
                    devices_in_AP_RO[m])
            trans_dispatch_process_delay.s_delay(
                N, M, devices, APs, tasks, devices_in_AP_RO, trans_rate, trans_time_RO, wireless_weight_RO, actual_rate_RO)
            trans_dispatch_process_delay.c_delay(
                M, N, APs, devices, tasks, devices_dispatched_RO, process_time_RO, CPU_weight_RO)
        else:
            for i in range(N):
                dispatch_time_RO[i] = 0
                process_time_RO[i] = 0
                for m in range(M):
                    trans_time_RO[i][m] = 0
                local_MD_RO.append(i+M)

                # ！！！！！！！！！！！！！！在Random offloading 中的卸载决定(MD随机选一个决定，迭代300次 +AP 不分派)！！！！！！！！！！！！！
                off_index_RO = random.randint(0, len(set(off_real_index[i]))-1)
                Off_decision_RO[i] = off_real_index[i][off_index_RO]

                if Off_decision_RO[i] < M:
                    dispatch_decision_RO[i] = Off_decision_RO[i]
                    local_MD_RO.remove(i+M)
                    devices_in_AP_RO[Off_decision_RO[i]].append(
                        i+M)  # 选择卸载到某AP的设备号
                    tasks_input_in_AP_RO[Off_decision_RO[i]].append(
                        tasks[i][1])  # 选择卸载到某AP的任务inputsize
                    tasks_cpu_in_AP_RO[Off_decision_RO[i]].append(
                        tasks[i][2])  # 选择卸载到某AP的任务cpucycles
            for m in range(M):
                for n in range(M):
                    devices_dispatched_RO[m][n].clear()
                devices_dispatched_RO[m][m] = copy.deepcopy(
                    devices_in_AP_RO[m])
            trans_dispatch_process_delay.s_delay(
                N, M, devices, APs, tasks, devices_in_AP_RO, trans_rate, trans_time_RO, wireless_weight_RO, actual_rate_RO)
            trans_dispatch_process_delay.c_delay(
                M, N, APs, devices, tasks, devices_dispatched_RO, process_time_RO, CPU_weight_RO)
            # trans_dispatch_process_delay.d_delay(N,M,tasks,devices_dispatched_RO,dispatch_time,dispatch_road_width,wired_weight_RO)
        for i in local_MD_RO:  # 本地执行的设备的时延存储
            total_latency_RO[i-M] = process_time_RO[i-M]
        for m in range(M):  # 初始dispatch决定带来的时延
            AP_latency_RO[m] = 0
            for i in devices_in_AP_RO[m]:
                # trans_dispatch_process_delay.d_delay(i,M,tasks,devices_dispatched_RO,dispatch_time_RO,dispatch_rate)
                # +dispatch_time_RO[i-M]+process_time_RO[i-M]
                total_latency_RO[i-M] = trans_time_RO[i -
                                                      M][m]+process_time_RO[i-M]
                AP_latency_RO[m] = AP_latency_RO[m]+total_latency_RO[i-M]
            # devices_dispatched[m][m]=[]    #清空列表
        leiji_total_latency_RO.append(sum(total_latency_RO))
        # print('AP_latency_RO',AP_latency_RO)
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Random offloading algorithm 结束!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # -----------------------------AP迭代------------------------------------------------------------------------------------------
        # 最开始的初始化dispatch decisions就是AP自己执行自己的，然后如果有AP能通过分派任务使得自身的任务时延减小，令这些AP中 \sum(CPUcyccles)/\sum(inputsiza)最大的AP先更新策略,比例要不要加期望要再想一下
        # tasks_input_cpu_ratio_rank=[[]for i in range(M)]  #\sum(CPUcyccles)/\sum(inputsiza)的存储列表
        tasks_input_cpu_ratio_rank_GO = [[]for i in range(M)]
        tasks_input_cpu_ratio_rank_FO = [[]for i in range(M)]

        # ============================开始AP的决策更新=(把对比方法的也同时算出来)=====================================
        for m in range(M):  # 计算并降序排序\sum(CPUcyccles)/\sum(inputsiza)
            '''if sum(tasks_input_in_AP_old[m])>0:
                tasks_input_cpu_ratio_rank[m]=(sum(tasks_cpu_in_AP_old[m])/APs[m][4])#sum(tasks_input_in_AP[m])
            else:
                tasks_input_cpu_ratio_rank[m]=0'''

            if sum(tasks_input_in_AP_GO[m]) > 0:
                tasks_input_cpu_ratio_rank_GO[m] = (
                    sum(tasks_cpu_in_AP_GO[m])/APs[m][4])  # sum(tasks_input_in_AP[m])
            else:
                tasks_input_cpu_ratio_rank_GO[m] = 0
            # ---------------------------------------------------------------------------------------------
            if sum(tasks_input_in_AP_FO[m]) > 0:
                tasks_input_cpu_ratio_rank_FO[m] = (
                    sum(tasks_cpu_in_AP_FO[m])/APs[m][4])  # sum(tasks_input_in_AP[m])
            else:
                tasks_input_cpu_ratio_rank_FO[m] = 0
            # tasks_input_cpu_ratio_rank_copy.append(sum(tasks_cpu_in_AP[m])/sum(tasks_input_in_AP[m]))

            # ---------------------------------------------------------------------------------------------
        # print(tasks_input_cpu_ratio_rank)
        # tasks_input_cpu_ratio_rank_copy=np.argsort(-np.array(tasks_input_cpu_ratio_rank))
        #
        tasks_input_cpu_ratio_rank_copy_GO = np.argsort(
            -np.array(tasks_input_cpu_ratio_rank_GO))
        tasks_input_cpu_ratio_rank_copy_FO = np.argsort(
            -np.array(tasks_input_cpu_ratio_rank_FO))
        #
        # print(tasks_input_cpu_ratio_rank_copy)
        # print("tasks_cpu_in_AP",tasks_cpu_in_AP)
        # ----------------------------------GO-------------------------------------------------------------
        while(True):  # 只要AP策略没收敛，就一直更新
            devices_dispatched_compare_copy_GO = copy.deepcopy(
                devices_dispatched_GO)  # ------------------
            for h in range(M):  # 按排序使AP异步更新
                # m_GO为\sum(CPUcyccles)/\sum(inputsiza)第h大的AP
                # ------------------------------------
                m_GO = tasks_input_cpu_ratio_rank_copy_GO[h]
                if tasks_input_cpu_ratio_rank_GO[m_GO] == 0:
                    break
                # print("轮到AP:",m_GO)
                # task_cpu_in_AP_copy_GO是降序排序的原列表从大到小的索引
                task_cpu_in_AP_copy_GO = np.argsort(
                    -np.array(tasks_cpu_in_AP_GO[m_GO]))
                # print("task_cpu_in_AP_copy_GO",task_cpu_in_AP_copy_GO)
                # print("devices_in_AP_GO[m_GO]",devices_in_AP_GO[m_GO])
                for j in range(len(set(devices_in_AP_GO[m_GO]))):
                    # print(task_cpu_in_AP_copy_GO[j])
                    # print("j=",j,"devices_in_AP_GO[m_GO]",devices_in_AP_GO[m_GO])
                    max_i_cpu_GO = task_cpu_in_AP_copy_GO[j]
                    # print("max_i_cpu_GO:",max_i_cpu_GO)
                    # 求出m_GO上cpu cycles最大的任务所属设备号
                    max_i_GO = devices_in_AP_GO[m_GO][max_i_cpu_GO]
                    # print("max_i_GO:",max_i_GO)
                    #print("off_decision_GO of",max_i_GO,":",Off_decision_GO[max_i_GO-M])
                    # 下面这一顿deepcopy是为了存储到最好的分派决策之后再更新原始的策略-----
                    dispatch_time_copy_GO = copy.deepcopy(dispatch_time_GO)
                    process_time_copy_GO = copy.deepcopy(process_time_GO)
                    CPU_weight_copy_GO = copy.deepcopy(CPU_weight_GO)
                    total_latency_copy_GO = copy.deepcopy(total_latency_GO)
                    AP_latency_copy_GO = copy.deepcopy(AP_latency_GO)
                    # Off_decision_copy_GO=copy.deepcopy(Off_decision_GO)
                    dispatch_decision_copy_GO = copy.deepcopy(
                        dispatch_decision_GO)
                    #print("dispatch_decision_GO  of",max_i_GO,":",dispatch_decision_GO[max_i_GO-M])
                    devices_dispatched_copy_GO = copy.deepcopy(
                        devices_dispatched_GO)  # -------------
                    MDs_in_path_between_APs_copy_GO = copy.deepcopy(
                        MDs_in_path_between_APs_GO)
                    # print("MDs_in_path_between_APs_GO",MDs_in_path_between_APs_GO)

                    for n in range(M):  # 循环，为max_i_GO挑出最好的修改分派方案
                        # print('n=',n)
                        # 下面这一顿deepcopy是使得每次计算针对当前max_i_GO的分派决定需要有好效果才更新到原有的copy版本-----
                        dispatch_time_copy2_GO = copy.deepcopy(
                            dispatch_time_copy_GO)
                        process_time_copy2_GO = copy.deepcopy(
                            process_time_copy_GO)
                        CPU_weight_copy2_GO = copy.deepcopy(CPU_weight_copy_GO)
                        total_latency_copy2_GO = copy.deepcopy(
                            total_latency_copy_GO)
                        AP_latency_copy2_GO = copy.deepcopy(AP_latency_copy_GO)
                        # Off_decision_copy2_GO=copy.deepcopy(Off_decision_copy_GO)
                        dispatch_decision_copy2_GO = copy.deepcopy(
                            dispatch_decision_copy_GO)
                        devices_dispatched_copy2_GO = copy.deepcopy(
                            devices_dispatched_copy_GO)  # -------------
                        wired_weight_copy2_GO = copy.deepcopy(wired_weight_GO)
                        MDs_in_path_between_APs_copy2_GO = copy.deepcopy(
                            MDs_in_path_between_APs_copy_GO)

                        # print("dispatch_decision_copy2_GO[",max_i_GO,"]",dispatch_decision_copy2_GO[max_i_GO-M])
                        # print("devices_dispatched_copy2_GO[",m_GO,"]",devices_dispatched_copy2_GO[m_GO])
                        # print("devices_dispatched_copy2_GO[",m_GO,"][",dispatch_decision_copy2_GO[max_i_GO-M],"]",devices_dispatched_copy2_GO[m_GO][dispatch_decision_copy2_GO[max_i_GO-M]])
                        # 把cpucycles最大的设备先从原分配方案中删去，然后换新的计算延迟
                        devices_dispatched_copy2_GO[m_GO][dispatch_decision_copy2_GO[max_i_GO-M]].remove(
                            max_i_GO)
                        # ————————————————————————2020-8-14——————虚拟移除原来分派选路上的MD——————————
                        # print(m_GO,dispatch_decision_copy2_GO[max_i_GO-M],'-=-',n)
                        # print(path_between_APs[m_GO][dispatch_decision_copy2_GO[max_i_GO-M]],max_i_GO)
                        # print(MDs_in_path_between_APs_copy2_GO)
                        for m_ in range(len(path_between_APs[m_GO][dispatch_decision_copy2_GO[max_i_GO-M]])-1):
                            if m_ != len(path_between_APs[m_GO][dispatch_decision_copy2_GO[max_i_GO-M]])-1:
                                # print(m_,path_between_APs[m_GO][dispatch_decision_copy2_GO[max_i_GO-M]][m_])
                                # print(path_between_APs[m_GO][dispatch_decision_copy2_GO[max_i_GO-M]][m_+1])
                                # print(MDs_in_path_between_APs_copy2_GO[path_between_APs[m_GO][dispatch_decision_copy2_GO[max_i_GO-M]][m_]][path_between_APs[m_GO][dispatch_decision_copy2_GO[max_i_GO-M]][m_+1]])
                                MDs_in_path_between_APs_copy2_GO[path_between_APs[m_GO][dispatch_decision_copy2_GO[max_i_GO-M]]
                                                                 [m_]][path_between_APs[m_GO][dispatch_decision_copy2_GO[max_i_GO-M]][m_+1]].remove(max_i_GO)
                        '''for m_ in range(len(path_between_APs[m][dispatch_decision_copy2_GO[max_i_GO-M]])-2):
                            MDs_in_path_between_APs_copy2_GO[m_][m_+1].remove(max_i_GO)'''
                        # ——————————————————————————————————————————————————————————————————————
                        dispatch_decision_copy2_GO[max_i_GO-M] = n
                        # 接下来计算把max_i_GO放到n上产生的时延
                        devices_dispatched_copy2_GO[m_GO][dispatch_decision_copy2_GO[max_i_GO-M]].append(
                            max_i_GO)
                        # trans_dispatch_process_delay.d_delay(max_i_GO,M,tasks,devices_dispatched_copy2_GO,dispatch_time_copy2_GO,dispatch_rate)
                        # ————————————————————————2020-7-17——————虚拟增加当前选择分派选路上的MD——————————
                        for m_ in range(len(path_between_APs[m_GO][n])-1):
                            if m_ != len(path_between_APs[m_GO][dispatch_decision_copy2_GO[max_i_GO-M]])-1:
                                MDs_in_path_between_APs_copy2_GO[path_between_APs[m_GO][dispatch_decision_copy2_GO[max_i_GO-M]]
                                                                 [m_]][path_between_APs[m_GO][dispatch_decision_copy2_GO[max_i_GO-M]][m_+1]].append(max_i_GO)
                        '''for m_ in range(len(path_between_APs[m][n])-2):
                            MDs_in_path_between_APs_copy2_GO[m_][m_+1].append(max_i_GO)'''
                        # ——————————————————————————————————————————————————————————————————————
                        trans_dispatch_process_delay.d_delay(N, M, tasks, devices_dispatched_copy2_GO, dispatch_time_copy2_GO,
                                                             dispatch_road_width, wired_weight_copy2_GO, path_between_APs, MDs_in_path_between_APs_copy2_GO)
                        trans_dispatch_process_delay.c_delay(
                            M, N, APs, devices, tasks, devices_dispatched_copy2_GO, process_time_copy2_GO, CPU_weight_copy2_GO)
                        AP_latency_try_GO = 0
                        devices_not_in_GO = []
                        for i in range(N):
                            devices_not_in_GO.append(i+M)
                        for i in devices_in_AP_GO[m_GO]:
                            devices_not_in_GO.remove(i)
                            total_latency_copy2_GO[i-M] = trans_time_GO[i-M][m_GO] + \
                                dispatch_time_copy2_GO[i-M] + \
                                process_time_copy2_GO[i-M]
                            AP_latency_try_GO = AP_latency_try_GO + \
                                total_latency_copy2_GO[i-M]

                        for i in devices_not_in_GO:
                            total_latency_copy2_GO[i-M] = sum(
                                trans_time_GO[i-M])+dispatch_time_copy2_GO[i-M]+process_time_copy2_GO[i-M]
                        # 如果这个选择是比原来的好，那就更新策略，保存时延结果
                        if AP_latency_try_GO < AP_latency_copy2_GO[m_GO]:

                            AP_latency_copy_GO[m_GO] = copy.deepcopy(
                                AP_latency_try_GO)
                            dispatch_time_copy_GO = copy.deepcopy(
                                dispatch_time_copy2_GO)
                            process_time_copy_GO = copy.deepcopy(
                                process_time_copy2_GO)
                            CPU_weight_copy_GO = copy.deepcopy(
                                CPU_weight_copy2_GO)
                            total_latency_copy_GO = copy.deepcopy(
                                total_latency_copy2_GO)
                            dispatch_decision_copy_GO = copy.deepcopy(
                                dispatch_decision_copy2_GO)
                            devices_dispatched_copy_GO = copy.deepcopy(
                                devices_dispatched_copy2_GO)
                            MDs_in_path_between_APs_copy_GO = copy.deepcopy(
                                MDs_in_path_between_APs_copy2_GO)

                    # 下面的deepcopy把针对与当前max_i_GO的最好结果进行更新
                    AP_latency_GO[m_GO] = copy.deepcopy(
                        AP_latency_copy_GO[m_GO])
                    dispatch_time_GO = copy.deepcopy(dispatch_time_copy_GO)
                    process_time_GO = copy.deepcopy(process_time_copy_GO)
                    CPU_weight_GO = copy.deepcopy(CPU_weight_copy_GO)
                    total_latency_GO = copy.deepcopy(total_latency_copy_GO)
                    dispatch_decision_GO = copy.deepcopy(
                        dispatch_decision_copy_GO)
                    devices_dispatched_GO = copy.deepcopy(
                        devices_dispatched_copy_GO)
                    MDs_in_path_between_APs_GO = copy.deepcopy(
                        MDs_in_path_between_APs_copy_GO)
            if devices_dispatched_GO == devices_dispatched_compare_copy_GO:
                # print("total_latency_GO",total_latency_GO)
                break

        # __________________________________FO______________________________________________________________
        while(True):  # 只要AP策略没收敛，就一直更新
            devices_dispatched_compare_copy_FO = copy.deepcopy(
                devices_dispatched_FO)  # ------------------
            for h in range(M):  # 按排序使AP异步更新
                # m_GO为\sum(CPUcyccles)/\sum(inputsiza)第h大的AP
                # ------------------------------------
                m_FO = tasks_input_cpu_ratio_rank_copy_FO[h]
                if tasks_input_cpu_ratio_rank_FO[m_FO] == 0:
                    break
                # task_cpu_in_AP_copy_FO是降序排序的原列表从大到小的索引
                task_cpu_in_AP_copy_FO = np.argsort(
                    -np.array(tasks_cpu_in_AP_FO[m_FO]))

                for j in range(len(set(devices_in_AP_FO[m_FO]))):
                    # print(task_cpu_in_AP_copy_FO[j])
                    max_i_cpu_FO = task_cpu_in_AP_copy_FO[j]
                    # print(max_i_cpu_FO)
                    # 求出m_FO上cpu cycles最大的任务所属设备号
                    max_i_FO = devices_in_AP_FO[m_FO][max_i_cpu_FO]
                    # print(max_i_FO)
                    # 下面这一顿deepcopy是为了存储到最好的分派决策之后再更新原始的策略-----
                    dispatch_time_copy_FO = copy.deepcopy(dispatch_time_FO)
                    process_time_copy_FO = copy.deepcopy(process_time_FO)
                    CPU_weight_copy_FO = copy.deepcopy(CPU_weight_FO)
                    total_latency_copy_FO = copy.deepcopy(total_latency_FO)
                    AP_latency_copy_FO = copy.deepcopy(AP_latency_FO)
                    # Off_decision_copy_FO=copy.deepcopy(Off_decision_FO)
                    dispatch_decision_copy_FO = copy.deepcopy(
                        dispatch_decision_FO)
                    devices_dispatched_copy_FO = copy.deepcopy(
                        devices_dispatched_FO)  # -------------
                    MDs_in_path_between_APs_copy_FO = copy.deepcopy(
                        MDs_in_path_between_APs_FO)
                    for n in range(M):  # 循环，为max_i_GO挑出最好的修改分派方案
                        # print("n:",n)
                        # 下面这一顿deepcopy是使得每次计算针对当前max_i_FO的分派决定需要有好效果才更新到原有的copy版本-----
                        dispatch_time_copy2_FO = copy.deepcopy(
                            dispatch_time_copy_FO)
                        process_time_copy2_FO = copy.deepcopy(
                            process_time_copy_FO)
                        CPU_weight_copy2_FO = copy.deepcopy(CPU_weight_copy_FO)
                        total_latency_copy2_FO = copy.deepcopy(
                            total_latency_copy_FO)
                        AP_latency_copy2_FO = copy.deepcopy(AP_latency_copy_FO)
                        # Off_decision_copy2_FO=copy.deepcopy(Off_decision_copy_FO)
                        dispatch_decision_copy2_FO = copy.deepcopy(
                            dispatch_decision_copy_FO)
                        devices_dispatched_copy2_FO = copy.deepcopy(
                            devices_dispatched_copy_FO)  # -------------
                        MDs_in_path_between_APs_copy2_FO = copy.deepcopy(
                            MDs_in_path_between_APs_copy_FO)

                        wired_weight_copy2_FO = copy.deepcopy(wired_weight_FO)

                        # print("dispatch_decision_copy2_FO[",max_i_FO,"]",dispatch_decision_copy2_FO[max_i_FO-M])
                        # print("devices_dispatched_copy2_FO[",m_FO,"]",devices_dispatched_copy2_FO[m_FO])
                        # print("devices_dispatched_copy2_FO[",m_FO,"][",dispatch_decision_copy2_FO[max_i_FO-M],"]",devices_dispatched_copy2_FO[m_FO][dispatch_decision_copy2_FO[max_i_FO-M]])

                        # 把cpucycles最大的设备先从原分配方案中删去，然后换新的计算延迟
                        devices_dispatched_copy2_FO[m_FO][dispatch_decision_copy2_FO[max_i_FO-M]].remove(
                            max_i_FO)

                        # print("m_FO",m_FO,dispatch_decision_copy2_FO[max_i_FO-M],'-=-',n)
                        # print(path_between_APs[m_FO][dispatch_decision_copy2_FO[max_i_FO-M]],max_i_FO)
                        # print(MDs_in_path_between_APs_copy2_FO)
                        # ————————————————————————2020-7-17——————虚拟移除原来分派选路上的MD——————————
                        for m_ in range(len(path_between_APs[m_FO][dispatch_decision_copy2_FO[max_i_FO-M]])-1):
                            if m_ != len(path_between_APs[m_FO][dispatch_decision_copy2_FO[max_i_FO-M]])-1:
                                # print(m_,path_between_APs[m_FO][dispatch_decision_copy2_FO[max_i_FO-M]][m_])
                                # print(path_between_APs[m_FO][dispatch_decision_copy2_FO[max_i_FO-M]][m_+1])
                                # print(MDs_in_path_between_APs_copy2_FO[path_between_APs[m_FO][dispatch_decision_copy2_FO[max_i_FO-M]][m_]][path_between_APs[m_FO][dispatch_decision_copy2_FO[max_i_FO-M]][m_+1]])

                                MDs_in_path_between_APs_copy2_FO[path_between_APs[m_FO][dispatch_decision_copy2_FO[max_i_FO-M]]
                                                                 [m_]][path_between_APs[m_FO][dispatch_decision_copy2_FO[max_i_FO-M]][m_+1]].remove(max_i_FO)
                        # ——————————————————————————————————————————————————————————————————————
                        dispatch_decision_copy2_FO[max_i_FO-M] = n
                        # 接下来计算把max_i_FO放到n上产生的时延
                        devices_dispatched_copy2_FO[m_FO][dispatch_decision_copy2_FO[max_i_FO-M]].append(
                            max_i_FO)
                        # print(path_between_APs[m_FO][dispatch_decision_copy2_FO[max_i_FO-M]])
                        # trans_dispatch_process_delay.d_delay(max_i_FO,M,tasks,devices_dispatched_copy2_FO,dispatch_time_copy2_FO,dispatch_rate)
                        # ————————————————————————2020-7-17——————虚拟增加当前选择分派选路上的MD——————————
                        for m_ in range(len(path_between_APs[m_FO][n])-1):
                            # print('??',path_between_APs[m_FO][dispatch_decision_copy2_FO[max_i_FO-M]][m_])
                            if m_ != len(path_between_APs[m_FO][dispatch_decision_copy2_FO[max_i_FO-M]])-1:
                                MDs_in_path_between_APs_copy2_FO[path_between_APs[m_FO][dispatch_decision_copy2_FO[max_i_FO-M]]
                                                                 [m_]][path_between_APs[m_FO][dispatch_decision_copy2_FO[max_i_FO-M]][m_+1]].append(max_i_FO)
                        # ——————————————————————————————————————————————————————————————————————
                        trans_dispatch_process_delay.d_delay(N, M, tasks, devices_dispatched_copy2_FO, dispatch_time_copy2_FO,
                                                             dispatch_road_width, wired_weight_copy2_FO, path_between_APs, MDs_in_path_between_APs_copy2_FO)
                        trans_dispatch_process_delay.c_delay(
                            M, N, APs, devices, tasks, devices_dispatched_copy2_FO, process_time_copy2_FO, CPU_weight_copy2_FO)
                        AP_latency_try_FO = 0
                        devices_not_in_FO = []
                        for i in range(N):
                            devices_not_in_FO.append(i+M)
                        for i in devices_in_AP_FO[m_FO]:
                            devices_not_in_FO.remove(i)
                            total_latency_copy2_FO[i-M] = trans_time_FO[i-M][m_FO] + \
                                dispatch_time_copy2_FO[i-M] + \
                                process_time_copy2_FO[i-M]
                            AP_latency_try_FO = AP_latency_try_FO + \
                                total_latency_copy2_FO[i-M]

                        for i in devices_not_in_FO:
                            total_latency_copy2_FO[i-M] = sum(
                                trans_time_FO[i-M])+dispatch_time_copy2_FO[i-M]+process_time_copy2_FO[i-M]
                        # 如果这个选择是比原来的好，那就更新策略，保存时延结果
                        if AP_latency_try_FO < AP_latency_copy2_FO[m_FO]:
                            AP_latency_copy_FO[m_FO] = copy.deepcopy(
                                AP_latency_try_FO)
                            dispatch_time_copy_FO = copy.deepcopy(
                                dispatch_time_copy2_FO)
                            process_time_copy_FO = copy.deepcopy(
                                process_time_copy2_FO)
                            CPU_weight_copy_FO = copy.deepcopy(
                                CPU_weight_copy2_FO)
                            total_latency_copy_FO = copy.deepcopy(
                                total_latency_copy2_FO)
                            dispatch_decision_copy_FO = copy.deepcopy(
                                dispatch_decision_copy2_FO)
                            devices_dispatched_copy_FO = copy.deepcopy(
                                devices_dispatched_copy2_FO)
                            MDs_in_path_between_APs_copy_FO = copy.deepcopy(
                                MDs_in_path_between_APs_copy2_FO)

                    # 下面的deepcopy把针对与当前max_i_GO的最好结果进行更新
                    AP_latency_FO[m_FO] = copy.deepcopy(
                        AP_latency_copy_FO[m_FO])
                    dispatch_time_FO = copy.deepcopy(dispatch_time_copy_FO)
                    process_time_FO = copy.deepcopy(process_time_copy_FO)
                    CPU_weight_FO = copy.deepcopy(CPU_weight_copy_FO)
                    total_latency_FO = copy.deepcopy(total_latency_copy_FO)
                    dispatch_decision_FO = copy.deepcopy(
                        dispatch_decision_copy_FO)
                    devices_dispatched_FO = copy.deepcopy(
                        devices_dispatched_copy_FO)
                    MDs_in_path_between_APs_FO = copy.deepcopy(
                        MDs_in_path_between_APs_copy_FO)
            if devices_dispatched_FO == devices_dispatched_compare_copy_FO:
                break
        # ____________________________________________________________________________________________________________________

        ave_round_delay_old

        # ----------------------------算AP利用率---------------------------
        # if t_d == MD_iteration_numbers-1:
        '''ave_max_delay_dif_GO += np.var(AP_latency_GO)#max(AP_latency_GO) - sum(AP_latency_GO)/M
        ave_offload_MD_num_GO += N-len(local_MD_GO)
        ave_APdelay_GO += sum(AP_latency_GO)

        ave_max_delay_dif_FO += np.var(AP_latency_FO)#max(AP_latency_FO) - sum(AP_latency_FO)/M
        ave_offload_MD_num_FO += N-len(local_MD_FO)
        ave_APdelay_FO += sum(AP_latency_FO)'''

        '''this_simu_ave_max_delay_dif_RO += np.var(AP_latency_RO)#max(AP_latency_RO) - sum(AP_latency_RO)/M
        this_simu_ave_max_delay_dif_RO = this_simu_ave_max_delay_dif_RO/MD_iteration_numbers
        ave_max_delay_dif_RO += this_simu_ave_max_delay_dif_RO
        #
        this_ave_offload_MD_num_RO += N-len(local_MD_RO)
        this_ave_offload_MD_num_RO = this_ave_offload_MD_num_RO/MD_iteration_numbers
        ave_offload_MD_num_RO += this_ave_offload_MD_num_RO
        #
        this_ave_APdelay_RO += sum(AP_latency_RO)
        this_ave_APdelay_RO = this_ave_APdelay_RO/MD_iteration_numbers
        #print("AP_latency_RO,this_ave_APdelay_RO",AP_latency_RO,this_ave_APdelay_RO)
        ave_APdelay_RO += this_ave_APdelay_RO'''

        if t_d == MD_iteration_numbers-1:
            print("AP_delay_wolf", AP_latency)
            print("AP_delay_old", AP_latency_old)
            print("AP_delay_GO", AP_latency_GO)
            print("AP_delay_FO", AP_latency_FO)
            print("AP_delay_DO", AP_latency_DO)
            print("AP_delay_RO", AP_latency_RO)
            # max(AP_latency_GO) - sum(AP_latency_GO)/M
            this_simu_ave_max_delay_dif_GO += np.var(AP_latency_GO)
            this_ave_offload_MD_num_GO += N-len(local_MD_GO)
            this_ave_APdelay_GO += sum(AP_latency_GO)

            # max(AP_latency_FO) - sum(AP_latency_FO)/M
            this_simu_ave_max_delay_dif_FO += np.var(AP_latency_FO)
            this_ave_offload_MD_num_FO += N-len(local_MD_FO)
            this_ave_APdelay_FO += sum(AP_latency_FO)

            ave_max_delay_dif_GO += this_simu_ave_max_delay_dif_GO / \
                MD_iteration_numbers  # max(AP_latency_GO) - sum(AP_latency_GO)/M
            ave_offload_MD_num_GO += this_ave_offload_MD_num_GO / MD_iteration_numbers
            ave_APdelay_GO += this_ave_APdelay_GO / MD_iteration_numbers

            ave_max_delay_dif_FO += this_simu_ave_max_delay_dif_FO / \
                MD_iteration_numbers  # max(AP_latency_FO) - sum(AP_latency_FO)/M
            ave_offload_MD_num_FO += this_ave_offload_MD_num_FO / MD_iteration_numbers
            ave_APdelay_FO += this_ave_APdelay_FO / MD_iteration_numbers

            # max(AP_latency_RO) - sum(AP_latency_RO)/M
            this_simu_ave_max_delay_dif_RO += np.var(AP_latency_RO)
            this_simu_ave_max_delay_dif_RO = this_simu_ave_max_delay_dif_RO/MD_iteration_numbers
            ave_max_delay_dif_RO += this_simu_ave_max_delay_dif_RO
            #
            this_ave_offload_MD_num_RO += N-len(local_MD_RO)
            this_ave_offload_MD_num_RO = this_ave_offload_MD_num_RO/MD_iteration_numbers
            ave_offload_MD_num_RO += this_ave_offload_MD_num_RO
            #
            this_ave_APdelay_RO += sum(AP_latency_RO)
            this_ave_APdelay_RO = this_ave_APdelay_RO/MD_iteration_numbers
            ave_APdelay_RO += this_ave_APdelay_RO

            '''this_simu_ave_max_delay_dif_DO += np.var(AP_latency_DO)#max(AP_latency_DO) - sum(AP_latency_DO)/M
            this_simu_ave_max_delay_dif_DO = this_simu_ave_max_delay_dif_DO/20
            ave_max_delay_dif_DO += this_simu_ave_max_delay_dif_DO
            #
            this_ave_offload_MD_num_DO += N-len(local_MD_DO)
            this_ave_offload_MD_num_DO = this_ave_offload_MD_num_DO/20
            ave_offload_MD_num_DO += this_ave_offload_MD_num_DO
            #
            this_ave_APdelay_DO += sum(AP_latency_DO)
            this_ave_APdelay_DO = this_ave_APdelay_DO/20
            #print("AP_latency_DO,this_ave_APdelay_DO",AP_latency_DO,this_ave_APdelay_DO)
            ave_APdelay_DO += this_ave_APdelay_DO'''

        '''if t_d >= MD_iteration_numbers-20 and t_d != MD_iteration_numbers-1:
            this_simu_ave_max_delay_dif_DO += np.var(AP_latency_DO)#max(AP_latency_DO) - sum(AP_latency_DO)/M
            this_ave_offload_MD_num_DO += N-len(local_MD_DO)
            this_ave_APdelay_DO += sum(AP_latency_DO)'''

        if t_d != MD_iteration_numbers-1:
            # max(AP_latency_GO) - sum(AP_latency_GO)/M
            this_simu_ave_max_delay_dif_GO += np.var(AP_latency_GO)
            this_ave_offload_MD_num_GO += N-len(local_MD_GO)
            this_ave_APdelay_GO += sum(AP_latency_GO)

            # max(AP_latency_FO) - sum(AP_latency_FO)/M
            this_simu_ave_max_delay_dif_FO += np.var(AP_latency_FO)
            this_ave_offload_MD_num_FO += N-len(local_MD_FO)
            this_ave_APdelay_FO += sum(AP_latency_FO)

            # max(AP_latency_RO) - sum(AP_latency_RO)/M
            this_simu_ave_max_delay_dif_RO += np.var(AP_latency_RO)
            this_ave_offload_MD_num_RO += N-len(local_MD_RO)
            this_ave_APdelay_RO += sum(AP_latency_RO)

            '''this_simu_ave_max_delay_dif_DO += np.var(AP_latency_DO)#max(AP_latency_DO) - sum(AP_latency_DO)/M
            this_ave_offload_MD_num_DO += N-len(local_MD_DO)
            this_ave_APdelay_DO += sum(AP_latency_DO)'''
        # -----------------------------------------------------------------

        # ----------------------------------------------GO------------------------------------------------------
        for i in range(N):
            # print(math.exp(device_Q_value[i][0]/lamda[i]))
            sum_Q_lamda_GO[i] = 0
            sum_Q_lamda_GO[i] = math.exp(device_Q_value_GO[i][0]/lamda[i])
            # sum_policy_GO=1
            for m in range(M):

                sum_Q_lamda_GO[i] = sum_Q_lamda_GO[i] + \
                    math.exp(device_Q_value_GO[i][m+1]/lamda[i])
            for m in range(M):
                if Off_decision_GO[i] == m:
                    device_Q_value_GO[i][m+1] = (
                        1-theda[i])*device_Q_value_GO[i][m+1]+theda[i]*(1/total_latency_GO[i])
                #print("sum_policy_GO of",i+M,":",sum_policy_GO)
            # device_policy_GO[i][0]=sum_policy_GO    #(math.exp(device_Q_value_GO[i][0]/lamda[i]))/sum_Q_lamda_GO[i]
            if Off_decision_GO[i] == i+M:
                device_Q_value_GO[i][0] = (
                    1-theda[i])*device_Q_value_GO[i][0]+theda[i]*(1/total_latency_GO[i])
            # print(device_policy_GO[i])

        # ============================================================================
        '''if t_d%50==0:
            print("--total_latency_old--:",total_latency_old)
            print("--device_policy_old--:",device_policy_old)'''
        # -------------------------------------------------------------------------------------------------------

    # print("--final_total_latency--::",total_latency)
    '''average_system_delay_wolf+=sum(total_latency)
    average_system_delay_old+=sum(total_latency_old)
    print("--sum_final_total_latency--:",sum(total_latency))
    print("--sum_final_total_latency_old--:",sum(total_latency_old))'''
    # print("--final_device_policy--:",device_policy)

    average_system_delay_wolf = ave_round_delay_wolf / \
        ave_round  # sum(leiji_total_latency_wolf)/20
    average_system_delay_old = ave_round_delay_old / \
        ave_round  # sum(leiji_total_latency_old)/20
    # average_system_delay_my=sum(leiji_total_latency)/30#(MD_iteration_numbers)#sum(total_latency)#average_system_delay_my+sum(total_latency)

    # sum(leiji_total_latency_GO)/(MD_iteration_numbers)#average_system_delay_GO+sum(total_latency_GO)
    average_system_delay_GO = sum(total_latency_GO)

    # sum(total_latency_FO)#average_system_delay_FO+sum(total_latency_FO)
    average_system_delay_FO = sum(
        leiji_total_latency_FO)/(MD_iteration_numbers)

    # sum(leiji_total_latency_DO)/(MD_iteration_numbers)#30#(MD_iteration_numbers)#sum(total_latency_DO)#average_system_delay_DO+sum(total_latency_DO)
    average_system_delay_DO = ave_round_delay_DO/ave_round

    # average_system_delay_RO+sum(leiji_total_latency_RO)/(MD_iteration_numbers)
    average_system_delay_RO = sum(
        leiji_total_latency_RO)/(MD_iteration_numbers)

    '''if 'N='+str(N)+',M='+str(M)+',compare_system_delay.csv' in os.listdir('F:\\小论文投稿\\Paper-2-边缘计算\\paper2-MEC-code-wolf-phc\\data_save'):
        with open('F:\\小论文投稿\\Paper-2-边缘计算\\paper2-MEC-code-wolf-phc\\data_save\\N='+str(N)+',M='+str(M)+',compare_system_delay.csv','a',newline='') as f:
            writer = csv.writer(f)
            writer.writerow([average_system_delay_wolf,average_system_delay_old, average_system_delay_GO, average_system_delay_FO,average_system_delay_DO, average_system_delay_RO])#,average_delay_per_AP,average_delay_per_AP_GO,average_delay_per_AP_FO,average_delay_per_AP_DO,average_delay_per_AP_RO,average_MD_numbers_per_AP,average_MD_numbers_per_AP_GO,average_MD_numbers_per_AP_FO,average_MD_numbers_per_AP_DO,average_MD_numbers_per_AP_RO])
    else:
        with open('F:\\小论文投稿\\Paper-2-边缘计算\\paper2-MEC-code-wolf-phc\\data_save\\N='+str(N)+',M='+str(M)+',compare_system_delay.csv','a',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['system_delay_wolf','system_delay_old', 'system_delay_GO', 'system_delay_FO','system_delay_DO', 'system_delay_RO'])#,'average_delay_per_AP_my','average_delay_per_AP_GO','average_delay_per_AP_FO','average_delay_per_AP_DO','average_delay_per_AP_RO','average_MD_numbers_per_AP_my','average_MD_numbers_per_AP_GO','average_MD_numbers_per_AP_FO','average_MD_numbers_per_AP_DO','average_MD_numbers_per_AP_RO'])
            writer.writerow([average_system_delay_wolf,average_system_delay_old, average_system_delay_GO, average_system_delay_FO,average_system_delay_DO, average_system_delay_RO])#,average_delay_per_AP,average_delay_per_AP_GO,average_delay_per_AP_FO,average_delay_per_AP_DO,average_delay_per_AP_RO,average_MD_numbers_per_AP,average_MD_numbers_per_AP_GO,average_MD_numbers_per_AP_FO,average_MD_numbers_per_AP_DO,average_MD_numbers_per_AP_RO])
    '''
    # 加了AP利用率部分的数据存储
    '''if 'N='+str(N)+',M='+str(M)+',compare_system_delay.csv' in os.listdir('F:\\小论文投稿\\Paper-2-边缘计算\\paper2-MEC-code-wolf-phc\\data_save'):
        with open('F:\\小论文投稿\\Paper-2-边缘计算\\paper2-MEC-code-wolf-phc\\data_save\\N='+str(N)+',M='+str(M)+',compare_system_delay.csv','a',newline='') as f:
            writer = csv.writer(f)
            writer.writerow([average_system_delay_wolf,average_system_delay_old, average_system_delay_GO, average_system_delay_FO,average_system_delay_DO, average_system_delay_RO,ave_max_delay_dif_wolf,ave_max_delay_dif_old,ave_simulation_delay_GO,ave_simulation_delay_FO,ave_simulation_delay_DO,ave_simulation_delay_RO])#,average_delay_per_AP,average_delay_per_AP_GO,average_delay_per_AP_FO,average_delay_per_AP_DO,average_delay_per_AP_RO,average_MD_numbers_per_AP,average_MD_numbers_per_AP_GO,average_MD_numbers_per_AP_FO,average_MD_numbers_per_AP_DO,average_MD_numbers_per_AP_RO])
    else:
        with open('F:\\小论文投稿\\Paper-2-边缘计算\\paper2-MEC-code-wolf-phc\\data_save\\N='+str(N)+',M='+str(M)+',compare_system_delay.csv','a',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['system_delay_wolf','system_delay_old', 'system_delay_GO', 'system_delay_FO','system_delay_DO', 'system_delay_RO','ave_max_delay_dif_wolf','ave_max_delay_dif_old','ave_simulation_delay_GO','ave_simulation_delay_FO','ave_simulation_delay_DO','ave_simulation_delay_RO'])#,'average_delay_per_AP_my','average_delay_per_AP_GO','average_delay_per_AP_FO','average_delay_per_AP_DO','average_delay_per_AP_RO','average_MD_numbers_per_AP_my','average_MD_numbers_per_AP_GO','average_MD_numbers_per_AP_FO','average_MD_numbers_per_AP_DO','average_MD_numbers_per_AP_RO'])
            writer.writerow([average_system_delay_wolf,average_system_delay_old, average_system_delay_GO, average_system_delay_FO,average_system_delay_DO, average_system_delay_RO,ave_max_delay_dif_wolf,ave_max_delay_dif_old,ave_simulation_delay_GO,ave_simulation_delay_FO,ave_simulation_delay_DO,ave_simulation_delay_RO])#,average_delay_per_AP,average_delay_per_AP_GO,average_delay_per_AP_FO,average_delay_per_AP_DO,average_delay_per_AP_RO,average_MD_numbers_per_AP,average_MD_numbers_per_AP_GO,average_MD_numbers_per_AP_FO,average_MD_numbers_per_AP_DO,average_MD_numbers_per_AP_RO])
    '''
    # 'x': simualtion_index,
    ave_simulation_delay_wolf += average_system_delay_wolf
    ave_simulation_delay_old += average_system_delay_old
    ave_simulation_delay_GO += average_system_delay_GO
    ave_simulation_delay_FO += average_system_delay_FO
    ave_simulation_delay_DO += average_system_delay_DO
    ave_simulation_delay_RO += average_system_delay_RO
    print("------------------------------第", simualtion_index +
          1, "次实验结束---------------------------------")


#sys_dy=sys_dy.append({'x': simualtion_index, 'system_delay_my': average_system_delay_my/simulation_numbers, 'system_delay_GO': average_system_delay_GO/simulation_numbers, 'system_delay_FO': average_system_delay_FO/simulation_numbers, 'system_delay_DO': average_system_delay_DO/simulation_numbers, 'system_delay_RO': average_system_delay_RO/simulation_numbers,'average_delay_per_AP_my':average_delay_per_AP/simulation_numbers,'average_delay_per_AP_GO':average_delay_per_AP_GO/simulation_numbers,'average_delay_per_AP_FO':average_delay_per_AP_FO/simulation_numbers,'average_delay_per_AP_DO':average_delay_per_AP_DO/simulation_numbers,'average_delay_per_AP_RO':average_delay_per_AP_RO/simulation_numbers,'average_MD_numbers_per_AP_my':average_MD_numbers_per_AP/simulation_numbers,'average_MD_numbers_per_AP_GO':average_MD_numbers_per_AP_GO/simulation_numbers,'average_MD_numbers_per_AP_FO':average_MD_numbers_per_AP_FO/simulation_numbers,'average_MD_numbers_per_AP_DO':average_MD_numbers_per_AP_DO/simulation_numbers,'average_MD_numbers_per_AP_RO':average_MD_numbers_per_AP_RO/simulation_numbers},ignore_index=True)
# dataname='F:\\小论文投稿\\Paper-2-边缘计算\\paper2-MEC-code\\data_save\\N='+str(N)+',M='+str(M)+'compare_system_delay'+str(m)+'.csv'
# sys_dy.to_csv(dataname)

print("====================参数设置  N=", N, ",M=", M,
      "时，system_delay的对比：=======================================")
print("--------------Wolf--------------system_delay :",
      ave_simulation_delay_wolf/simulation_numbers)
print("--------------Q-based old--------------system_delay :",
      ave_simulation_delay_old/simulation_numbers)
print("--------------GO--------------system_delay :",
      ave_simulation_delay_GO/simulation_numbers)
print("--------------FO--------------system_delay :",
      ave_simulation_delay_FO/simulation_numbers)
print("--------------DO--------------system_delay :",
      ave_simulation_delay_DO/simulation_numbers)
print("--------------RO--------------system_delay :",
      ave_simulation_delay_RO/simulation_numbers)
print("----------------------------------------------------------------------------------------------------------------------------------")
'''print("====================参数设置  N=",N,",M=",M,"时，(APdelay,卸载任务数)：=======================================")
print("--------------Wolf--------------:",ave_APdelay_wolf,ave_offload_MD_num_wolf)
print("--------------Q-based old-------------- :",ave_APdelay_old,ave_offload_MD_num_old)
print("--------------GO-------------- :",ave_APdelay_GO,ave_offload_MD_num_GO)
print("--------------FO-------------- :",ave_APdelay_FO,ave_offload_MD_num_FO)
print("--------------DO-------------- :",ave_APdelay_DO,ave_offload_MD_num_DO)
print("--------------RO-------------- :",ave_APdelay_RO,ave_offload_MD_num_RO)
print("----------------------------------------------------------------------------------------------------------------------------------")
print("====================参数设置  N=",N,",M=",M,"时，(AP方差)：=======================================")
print("--------------Wolf--------------:",ave_max_delay_dif_wolf)
print("--------------Q-based old--------------:",ave_max_delay_dif_old)
print("--------------GO--------------:",ave_max_delay_dif_GO)
print("--------------FO--------------:",ave_max_delay_dif_FO)
print("--------------DO--------------:",ave_max_delay_dif_DO)
print("--------------RO--------------:",ave_max_delay_dif_RO)
print("----------------------------------------------------------------------------------------------------------------------------------")
print("====================参数设置  N=",N,",M=",M,"时，(APdelay/totaldelay)：=======================================")
print("--------------Wolf-------------- :",ave_APdelay_wolf/ave_simulation_delay_wolf)
print("--------------Q-based old--------------:",ave_APdelay_old/ave_simulation_delay_old)
print("--------------GO--------------:",ave_APdelay_GO/ave_simulation_delay_GO)
print("--------------FO--------------:",ave_APdelay_FO/ave_simulation_delay_FO)
print("--------------DO--------------:",ave_APdelay_DO/ave_simulation_delay_DO)
print("--------------RO--------------:",ave_APdelay_RO/ave_simulation_delay_RO)
print("====================参数设置  N=",N,",M=",M,"时，(localdelay)：=======================================")
print("--------------Wolf-------------- :",ave_simulation_delay_wolf-ave_APdelay_wolf)
print("--------------Q-based old--------------:",ave_simulation_delay_old-ave_APdelay_old)
print("--------------GO--------------:",ave_simulation_delay_GO-ave_APdelay_GO)
print("--------------FO--------------:",ave_simulation_delay_FO-ave_APdelay_FO)
print("--------------DO--------------:",ave_simulation_delay_DO-ave_APdelay_DO)
print("--------------RO--------------:",ave_simulation_delay_RO-ave_APdelay_RO)'''
print("----------------------------------------------------------------------------------------------------------------------------------")
'''print("====================参数设置  N=",N,",M=",M,"时，AP利用率的对比-1(卸载任务数*APdelay/totaldelay/方差)：=======================================")
print("--------------Wolf-------------- :",ave_offload_MD_num_wolf*ave_APdelay_wolf/ave_simulation_delay_wolf/ave_max_delay_dif_wolf)
print("--------------Q-based old--------------:",ave_offload_MD_num_old*ave_APdelay_old/ave_simulation_delay_old/ave_max_delay_dif_old)
print("--------------GO--------------:",ave_offload_MD_num_GO*ave_APdelay_GO/ave_simulation_delay_GO/ave_max_delay_dif_GO)
print("--------------FO--------------:",ave_offload_MD_num_FO*ave_APdelay_FO/ave_simulation_delay_FO/ave_max_delay_dif_FO)
print("--------------DO--------------:",ave_offload_MD_num_DO*ave_APdelay_DO/ave_simulation_delay_DO/ave_max_delay_dif_DO)
print("--------------RO--------------:",ave_offload_MD_num_RO*ave_APdelay_RO/ave_simulation_delay_RO/ave_max_delay_dif_RO)'''

print("====================参数设置  N=", N, ",M=", M,
      "时，AP利用率的对比-2(卸载任务数*APdelay/totaldelay/方差/localdelay)：=======================================")
print("--------------Wolf-------------- :", ave_offload_MD_num_wolf*ave_APdelay_wolf /
      ave_simulation_delay_wolf/ave_max_delay_dif_wolf/(ave_simulation_delay_wolf-ave_APdelay_wolf))
print("--------------Q-based old--------------:", ave_offload_MD_num_old*ave_APdelay_old /
      ave_simulation_delay_old/ave_max_delay_dif_old/(ave_simulation_delay_old-ave_APdelay_old))
print("--------------GO--------------:", ave_offload_MD_num_GO*ave_APdelay_GO /
      ave_simulation_delay_GO/ave_max_delay_dif_GO/(ave_simulation_delay_GO-ave_APdelay_GO))
print("--------------FO--------------:", ave_offload_MD_num_FO*ave_APdelay_FO /
      ave_simulation_delay_FO/ave_max_delay_dif_FO/(ave_simulation_delay_FO-ave_APdelay_FO))
print("--------------DO--------------:", ave_offload_MD_num_DO*ave_APdelay_DO /
      ave_simulation_delay_DO/ave_max_delay_dif_DO/(ave_simulation_delay_DO-ave_APdelay_DO))
print("--------------RO--------------:", ave_offload_MD_num_RO*ave_APdelay_RO /
      ave_simulation_delay_RO/ave_max_delay_dif_RO/(ave_simulation_delay_RO-ave_APdelay_RO))

'''print("====================参数设置  N=",N,",M=",M,"时，AP利用率的对比-2(卸载任务数*APdelay/totaldelay)：=======================================")
print("--------------Wolf-------------- :",ave_offload_MD_num_wolf*ave_APdelay_wolf/ave_simulation_delay_wolf)
print("--------------Q-based old--------------:",ave_offload_MD_num_old*ave_APdelay_old/ave_simulation_delay_old)
print("--------------GO--------------:",ave_offload_MD_num_GO*ave_APdelay_GO/ave_simulation_delay_GO)
print("--------------FO--------------:",ave_offload_MD_num_FO*ave_APdelay_FO/ave_simulation_delay_FO)
print("--------------DO--------------:",ave_offload_MD_num_DO*ave_APdelay_DO/ave_simulation_delay_DO)
print("--------------RO--------------:",ave_offload_MD_num_RO*ave_APdelay_RO/ave_simulation_delay_RO)'''

'''print("====================参数设置  N=",N,",M=",M,"时，AP利用率的对比-2(卸载任务数*(APdelay/localdelay)/totaldelay)：=======================================")
print("--------------Wolf-------------- :",ave_offload_MD_num_wolf*(ave_APdelay_wolf/(ave_simulation_delay_wolf-ave_APdelay_wolf))/ave_simulation_delay_wolf)
print("--------------Q-based old--------------:",ave_offload_MD_num_old*(ave_APdelay_old/(ave_simulation_delay_old-ave_APdelay_old))/ave_simulation_delay_old)
print("--------------GO--------------:",ave_offload_MD_num_GO*(ave_APdelay_GO/(ave_simulation_delay_GO-ave_APdelay_GO))/ave_simulation_delay_GO)
print("--------------FO--------------:",ave_offload_MD_num_FO*(ave_APdelay_FO/(ave_simulation_delay_FO-ave_APdelay_FO))/ave_simulation_delay_FO)
print("--------------DO--------------:",ave_offload_MD_num_DO*(ave_APdelay_DO/(ave_simulation_delay_DO-ave_APdelay_DO))/ave_simulation_delay_DO)
print("--------------RO--------------:",ave_offload_MD_num_RO*(ave_APdelay_RO/(ave_simulation_delay_RO-ave_APdelay_RO))/ave_simulation_delay_RO)

print("====================参数设置  N=",N,",M=",M,"时，AP利用率的对比-2(localdelay*方差)：=======================================")
print("--------------Wolf-------------- :",(ave_simulation_delay_wolf-ave_APdelay_wolf)*ave_max_delay_dif_wolf/simulation_numbers**2)
print("--------------Q-based old--------------:",(ave_simulation_delay_old-ave_APdelay_old)*ave_max_delay_dif_old/simulation_numbers**2)
print("--------------GO--------------:",(ave_simulation_delay_GO-ave_APdelay_GO)*ave_max_delay_dif_GO/simulation_numbers**2)
print("--------------FO--------------:",(ave_simulation_delay_FO-ave_APdelay_FO)*ave_max_delay_dif_FO/simulation_numbers**2)
print("--------------DO--------------:",(ave_simulation_delay_DO-ave_APdelay_DO)*ave_max_delay_dif_DO/simulation_numbers**2)
print("--------------RO--------------:",(ave_simulation_delay_RO-ave_APdelay_RO)*ave_max_delay_dif_RO/simulation_numbers**2)
'''
'''print("====================参数设置  N=",N,",M=",M,"时，AP利用率的对比-2(localdelay*方差/Apdelay/(APdelay/totaldelay))：=======================================")
print("--------------Wolf-------------- :",(ave_simulation_delay_wolf-ave_APdelay_wolf)*ave_max_delay_dif_wolf/simulation_numbers/ave_APdelay_wolf/(ave_APdelay_wolf/ave_simulation_delay_wolf))
print("--------------Q-based old--------------:",(ave_simulation_delay_old-ave_APdelay_old)*ave_max_delay_dif_old/simulation_numbers/ave_APdelay_old/(ave_APdelay_old/ave_simulation_delay_old))
print("--------------GO--------------:",(ave_simulation_delay_GO-ave_APdelay_GO)*ave_max_delay_dif_GO/simulation_numbers/ave_APdelay_GO/(ave_APdelay_GO/ave_simulation_delay_GO))
print("--------------FO--------------:",(ave_simulation_delay_FO-ave_APdelay_FO)*ave_max_delay_dif_FO/simulation_numbers/ave_APdelay_FO/(ave_APdelay_FO/ave_simulation_delay_FO))
print("--------------DO--------------:",(ave_simulation_delay_DO-ave_APdelay_DO)*ave_max_delay_dif_DO/simulation_numbers/ave_APdelay_DO/(ave_APdelay_DO/ave_simulation_delay_DO))
print("--------------RO--------------:",(ave_simulation_delay_RO-ave_APdelay_RO)*ave_max_delay_dif_RO/simulation_numbers/ave_APdelay_RO/(ave_APdelay_RO/ave_simulation_delay_RO))
'''
