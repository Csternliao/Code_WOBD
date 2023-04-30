import copy
import math

import numpy as np
import random

# Parent Class(Device)


class DeviceAgent(object):

    def __init__(self, m, access_aps):
        self.M = m
        self.access_aps = [access_ap[0] for access_ap in access_aps]
        self.policy = self.get_init_policy()
        self.off_decision_space, self.off_real_space = self.get_off_space()
        self.off_decision = 0
        self.policy_history = [self.policy]

    def get_init_policy(self):
        num_access_aps = len(self.access_aps)
        avg_pro = 1 / (num_access_aps + 1)
        device_policy = [0 for _ in range(self.M+1)]
        device_policy[0] = avg_pro
        if len(self.access_aps) == 0:
            device_policy[0] = 1
        for ap in self.access_aps:
            device_policy[ap+1] = avg_pro

        return device_policy

    def get_off_space(self):
        off_dicision_space = list(range(self.M + 1))
        off_real_space = [0]
        for ap in self.access_aps:
            off_real_space.append(ap+1)

        return off_dicision_space, off_real_space

    def take_action(self):
        off_decision = np.random.choice(
            self.off_decision_space, p=self.policy)
        self.off_decision = off_decision

        return off_decision


# WoLFPHC: Subclass of DeviceAgent
class WoLFPHC(DeviceAgent):

    def __init__(self, M, access_aps, theta, s_delta_win, s_delta_loss):
        super().__init__(M, access_aps)
        self.Q = [0 for _ in range(self.M+1)]
        self.theta = theta    # Equ 13 0.1
        self.s_delta = 0
        self.s_delta_win, self.s_delta_loss = s_delta_win, s_delta_loss

        self.toal_latency = 0
        self.accumulated_total_latency = []
        self.converge = False
        self.game_history = [1 for _ in range(self.M+1)]

        self.avg_policy = copy.deepcopy(self.policy)

    def take_action(self):
        off_decision = np.random.choice(
            self.off_decision_space, p=self.policy)
        # print(i+M,"offloading_decision:",Off_decision[i])
        self.game_history[off_decision] += 1
        self.off_decision = off_decision

        return off_decision

    def update_avg_policy(self):
        for m in range(self.M + 1):
            if m in self.off_real_space:
                self.avg_policy[m] += 1 / (self.game_history[m]) * \
                    (self.policy[m] - self.avg_policy[m])

    def update_policy(self, offload_decision, total_delay):

        # 更新Q值
        self.Q[offload_decision] = (
            1-self.theta) * self.Q[offload_decision] + self.theta * (1 / total_delay)

        # 计算平均策略期望值
        def func(x, y): return x*y
        result = map(func, self.avg_policy, self.Q)
        list_result_a = list(result)
        sum_ave_value = sum(list_result_a)
        # 计算当前策略期望值
        resultc = map(func, self.policy, self.Q)
        list_result_c = list(resultc)
        sum_current_value = sum(list_result_c)

        # 比较平均策略和当前策略
        if sum_ave_value <= sum_current_value:
            # 用小参数s_delta_win
            self.s_delta = self.s_delta_win
        else:
            # 用大参数s_delta_lose
            self.s_delta = self.s_delta_loss

        Q_rank = np.argsort(-np.array(self.Q))
        max_delta = 0
        for num in range(self.M + 1):
            if num == 0 or num-1 in self.access_aps:
                if num == Q_rank[0]:
                    max_num = num
                else:
                    delta = - \
                        min(self.policy[num], self.s_delta /
                            (len(self.off_real_space)-1))
                    max_delta -= delta
                    self.policy[num] = self.policy[num] + delta
        self.policy[max_num] += max_delta
        self.policy_history.append(self.policy)
        # print(self.policy)


class QLearningAgent(DeviceAgent):
    def __init__(self, M, access_aps, lamda, theta, mode) -> None:
        super().__init__(M, access_aps)
        self.Q = [0 for _ in range(self.M+1)]
        self.lamda = lamda    # Equ 12 0.1
        self.theta = theta    # Equ 13 0.1
        self.mode = mode    # 'greedy' or 'mix'
        self.epsilon = 1
        self.decay = 0.95
        self.epsilon_min = 0.01
        self.first = True

    def update_policy(self, offload_decision, total_latency):
        # print(math.exp(device_Q_value[i][0]/lamda[i]))'
        self.Q[offload_decision] = (
            1-self.theta)*self.Q[offload_decision]+self.theta*(1/total_latency)
        sum_Q_lamda = 0
        self.policy.clear()
        # self.policy.append(0)
        for m in range(self.M+1):
            # print(math.exp(device_Q_value[i][m+1]/lamda[i]))
            if m == 0 or m-1 in self.access_aps:
                sum_Q_lamda += math.exp(self.Q[m]/self.lamda)
        # print("sum_Q_lamda[",i+M,"]",sum_Q_lamda[i])
        for m in range(self.M + 1):
            if m == 0 or m-1 in self.access_aps:
                x = (math.exp(self.Q[m]/self.lamda))/sum_Q_lamda
                self.policy.append(x)
            else:
                self.policy.append(0)

    def take_action(self):
        if self.mode == 'mix':
            off_decision = np.random.choice(
                self.off_decision_space, p=self.policy)
            # print(i+M,"offloading_decision:",Off_decision[i])
            self.off_decision = off_decision
        elif self.mode == 'greedy':
            # if self.first:
            #     off_decision = random.choice(self.off_real_space)
            #     self.first = False
            # else:
            #     off_decision = np.argmax(self.Q)
            #     print(off_decision)
            if random.random() < self.epsilon:
                off_decision = random.choice(self.off_real_space)
            else:
                off_decision = np.argmax(self.Q)
                if off_decision not in self.off_real_space:
                    raise IndexError
            self.epsilon = max(self.epsilon * self.decay, self.epsilon_min)
            self.off_decision = off_decision
        else:
            raise ImportError

        return off_decision


class DispatchAgent(object):

    def __init__(self, M, env, is_dispatch) -> None:
        self.M = M
        self.devices_dispatched = [[[] for i in range(M)] for i in range(M)]
        self.env = env
        self.tasks_input_cpu_ratio, self.tasks_input_cpu_ratio_rank = [
            0 for _ in range(self.M)], [0 for _ in range(self.M)]
        self.is_dispatch = is_dispatch

    def get_init_devices_dispatched(self, devices_dispatched):
        self.devices_dispatched = devices_dispatched

    def get_dispatch_decision(self):
        dispatch_decision = [
            device.agent.off_decision for device in self.env.devices]
        return dispatch_decision

    def get_init_ap_latency(self):
        ap_latency = [0 for _ in range(self.M)]
        for m in range(self.M):  # 初始dispatch决定带来的时延
            ap = self.env.aps[m]
            for device in ap.devices_in_ap:
                ap_latency[m] = ap_latency[m] + device.total_delay

        return ap_latency

    def get_tasks_input_cpu_ratio_rank(self):
        tasks_input_cpu_ratio = [0 for _ in range(self.M)]
        for m in range(self.M):  # 计算并排序\sum(CPUcyccles)/\sum(inputsiza)
            if sum(self.env.aps[m].tasks_input_in_ap) > 0:
                tasks_input_cpu_ratio[m] = (
                    sum(self.env.aps[m].tasks_input_in_ap) / self.env.aps[m].total_f)  # sum(tasks_input_in_AP[m])
            else:
                tasks_input_cpu_ratio[m] = 0
            # tasks_input_cpu_ratio_rank_copy.append(sum(tasks_cpu_in_AP[m])/sum(tasks_input_in_AP[m]))
        tasks_input_cpu_ratio_rank = np.argsort(
            -np.array(tasks_input_cpu_ratio))

        return tasks_input_cpu_ratio, tasks_input_cpu_ratio_rank

    def take_action(self, devices_dispatched):
        # 不分派
        if not self.is_dispatch:
            return devices_dispatched, self.get_dispatch_decision()

        self.tasks_input_cpu_ratio, self.tasks_input_cpu_ratio_rank = self.get_tasks_input_cpu_ratio_rank()
        dispatch_decision = self.get_dispatch_decision()
        mds_in_path_between_aps = self.env.get_mds_in_path_between_aps(
            devices_dispatched)
        ap_latency = self.get_init_ap_latency()
        # print(ap_latency)

        while(True):
            devices_dispatched_compare_copy = copy.deepcopy(
                devices_dispatched)
            for h in range(self.M):  # 按排序使AP异步更新
                # m为\sum(CPUcyccles)/\sum(inputsiza)第h大的AP
                m = self.tasks_input_cpu_ratio_rank[h]
                if self.tasks_input_cpu_ratio[m] == 0:
                    break
                ap = self.env.aps[m]
                # print("AP",m,"在更新")
                # print("tasks_cpu_in_AP[",m,"]",tasks_cpu_in_AP[m])
                # task_cpu_in_AP_copy=copy.deepcopy(tasks_cpu_in_AP[m])
                # task_cpu_in_AP_copy是降序排序的原列表从大到小的索引
                task_cpu_in_ap_rank = np.argsort(
                    np.array(ap.tasks_cpu_in_ap))
                # print(task_cpu_in_AP_copy)

                # devices_dispatched_compare_copy=copy.deepcopy(devices_dispatched)
                # print("devices_dispatched_compare_copy in h=",h,":",devices_dispatched_compare_copy)
                for j in range(len(set(ap.devices_in_ap))):
                    max_i_cpu = task_cpu_in_ap_rank[j]
                    # 求出m上cpu cycles最大的任务所属设备号
                    max_i = ap.devices_in_ap[max_i_cpu].id
                    ap_latency_copy = copy.deepcopy(ap_latency)
                    dispatch_decision_copy = copy.deepcopy(dispatch_decision)
                    mds_in_path_between_aps_copy = copy.deepcopy(
                        mds_in_path_between_aps)
                    devices_dispatched_copy = copy.deepcopy(devices_dispatched)
                    for n in range(self.M):  # 循环，为max_i挑出最好的修改分派方案 and n!=m
                        # 下面这一顿deepcopy是使得每次计算针对当前max_i的分派决定需要有好效果才更新到原有的copy版本-----
                        ap_latency_copy1 = copy.deepcopy(ap_latency_copy)
                        dispatch_decision_copy1 = copy.deepcopy(
                            dispatch_decision_copy)
                        mds_in_path_between_aps_copy1 = copy.deepcopy(
                            mds_in_path_between_aps_copy)
                        devices_dispatched_copy1 = copy.deepcopy(
                            devices_dispatched_copy)

                        # 把cpucycles最大的设备先从原分配方案中删去，然后换新的计算延迟
                        devices_dispatched_copy1[m][dispatch_decision_copy1[max_i]-1].remove(
                            max_i)
                        # ————————————————————————2020-7-17——————虚拟移除原来分派选路上的MD——————————
                        # print(m,dispatch_decision_copy2[max_i-M],'-=-',n)
                        # print(path_between_APs[m][dispatch_decision_copy2[max_i-M]])
                        for m_ in range(len(self.env.path_between_aps[m][dispatch_decision_copy1[max_i]-1])-1):
                            if m_ != len(self.env.path_between_aps[m][dispatch_decision_copy1[max_i]-1])-1:
                                # print(m_,path_between_APs[m][dispatch_decision_copy2[max_i-M]][m_],path_between_APs[m][dispatch_decision_copy2[max_i-M]][m_+1],MDs_in_path_between_APs_copy2[path_between_APs[m][dispatch_decision_copy2[max_i-M]][m_]][path_between_APs[m][dispatch_decision_copy2[max_i-M]][m_+1]],max_i)
                                mds_in_path_between_aps_copy1[self.env.path_between_aps[m][dispatch_decision_copy1[max_i]-1]
                                                              [m_]][self.env.path_between_aps[m][dispatch_decision_copy1[max_i]-1][m_+1]].remove(max_i)
                        # —————————————————————————————————————————————————————————————————————————
                        dispatch_decision_copy1[max_i] = n + 1
                        # 接下来计算把max_i放到n上产生的时延
                        devices_dispatched_copy1[m][dispatch_decision_copy1[max_i]-1].append(
                            max_i)
                        # trans_dispatch_process_delay.d_delay(max_i,M,tasks,devices_dispatched_copy2,dispatch_time_copy2,dispatch_rate)
                        # ————————————————————————2020-7-17——————虚拟更新当前选择分派路上的MD—————————————————————————————————
                        for m_ in range(len(self.env.path_between_aps[m][n])-1):
                            if m_ != len(self.env.path_between_aps[m][dispatch_decision_copy1[max_i] - 1])-1:
                                mds_in_path_between_aps_copy1[self.env.path_between_aps[m][dispatch_decision_copy1[max_i]-1]
                                                              [m_]][self.env.path_between_aps[m][dispatch_decision_copy1[max_i]-1][m_+1]].append(max_i)
                        # ——————————————————————————————————————————————————————————————————————
                        dispatch_delay = self.env.compute_dispatch_delay(
                            devices_dispatched_copy1)
                        ap_latency_try = 0
                        devices_not_in = []
                        total_latency = [0 for _ in range(self.env.N)]
                        for i in range(self.env.N):
                            devices_not_in.append(i)
                        for device in ap.devices_in_ap:
                            devices_not_in.remove(device.id)
                            # print("----------before copy----------")
                            # print("total_latency_copy2[",i,"]",total_latency_copy2[i-M])
                            divide_rule = 0
                            for x in range(self.M):
                                for i in devices_dispatched_copy1[x][dispatch_decision_copy1[device.id]-1]:
                                    divide_rule += math.sqrt(
                                        device.task[1] / self.env.aps[dispatch_decision_copy1[device.id]-1].total_f)
                            total_latency[device.id] = device.trans_delay + \
                                dispatch_delay[device.id] + \
                                self.env.compute_process_delay(
                                    device, self.env.aps[dispatch_decision_copy1[device.id]-1], is_ap=True, divide_rule=divide_rule)
                            #print("----------after copy----------")
                            # print("total_latency_copy2[",i,"]",total_latency_copy2[i-M])
                            # Ap_lateny_copy[m]=0
                            ap_latency_try = ap_latency_try + \
                                total_latency[device.id]
                            # AP_latency_copy[m]=AP_latency_copy[m]+total_latency_copy[i]
                        # print("AP",m,"_latency_try:",AP_latency_try)
                        # print("AP",m,"_latency_copy2:",(AP_latency_copy2[m]))
                        # 如果这个选择是比原来的好，那就更新策略，保存时延结果
                        # print(ap_latency)
                        if ap_latency_try < ap_latency_copy1[m]:
                            ap_latency_copy[m] = copy.deepcopy(ap_latency_try)
                            dispatch_decision_copy = copy.deepcopy(
                                dispatch_decision_copy1)
                            devices_dispatched_copy = copy.deepcopy(
                                devices_dispatched_copy1)
                            mds_in_path_between_aps_copy = copy.deepcopy(
                                mds_in_path_between_aps_copy1)

                    ap_latency[m] = copy.deepcopy(ap_latency_copy[m])
                    dispatch_decision = copy.deepcopy(dispatch_decision_copy)
                    devices_dispatched = copy.deepcopy(devices_dispatched_copy)
                    mds_in_path_between_aps = copy.deepcopy(
                        mds_in_path_between_aps_copy)
            #print("AP ",m,"更新后的分派情况:",devices_dispatched)
            if devices_dispatched == devices_dispatched_compare_copy:
                break

        return devices_dispatched, dispatch_decision
