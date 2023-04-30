"""
    Model of Device、AP
"""

import math

import networkx as nx

from method import *
from config import *


class Device(object):

    def __init__(self, id, loc: list[int], f_c=None, f_p=None, task: list[int]=None, group=None) -> None:
        self.id = id
        self.loc = loc  # location [x, y]
        self.f_c = f_c  # cpu frequency
        self.f_p = f_p  # trans frequency
        self.task = task    # info of task [input_size, cpu_cycles]
        self.access_aps = []  # linked ap(given by env) [ap.id, trans_rate]

        self.group = group

        self.trans_delay = 0
        self.dispatch_delay = 0
        self.process_delay = 0
        self.total_delay = 0

        self.agent: DeviceAgent = None

    def link_agent(self, agent):
        self.agent = agent

    def get_trans_rate(self, ap_id):
        for aid, trans_rate in self.access_aps:
            if aid == ap_id:
                return trans_rate
        raise ValueError

    def update_delay(self, trans_delay, dispatch_delay, process_delay):
        self.trans_delay, self.dispatch_delay, self.process_delay = trans_delay, dispatch_delay, process_delay
        self.total_delay = self.trans_delay + self.dispatch_delay + self.process_delay


class AP(object):

    def __init__(self, id, total_b=None, total_f=None, loc=None, group=None) -> None:
        self.id = id
        self.loc = loc  # location (x, y)
        self.total_b = total_b
        self.total_f = total_f

        self.group = group

        self.devices_in_ap = []
        self.tasks_input_in_ap = []
        self.tasks_cpu_in_ap = []

        self.devices_in_ap_dispatched = []

        self.dispatch_time = None

    def reset(self):
        self.devices_in_ap.clear()
        self.tasks_cpu_in_ap.clear()
        self.tasks_input_in_ap.clear()
        self.devices_in_ap_dispatched.clear()

    def get_bandwidth_divide_rule(self):
        bandwidth_divide_rule = 0
        for device in self.devices_in_ap:
            bandwidth_divide_rule += math.sqrt(
                device.task[0] / device.get_trans_rate(self.id))

        return bandwidth_divide_rule

    def get_cpu_divide_rule(self):
        cpu_divide_rule = 0
        for device in self.devices_in_ap_dispatched:
            cpu_divide_rule += math.sqrt(device.task[1] / self.total_f)

        return cpu_divide_rule


class Env(object):
    def __init__(self, N, M, devices: list[Device], aps: list[AP], ap_matrix, wired_width, path_loss, noise) -> None:
        self.N = N
        self.M = M
        self.devices = devices
        self.aps = aps
        self.ap_matrix = ap_matrix
        self.wired_width = wired_width
        self.path_loss = path_loss
        self.noise = noise

        self.get_access_aps()
        self.path_between_aps = self.get_path_between_aps()

    def get_access_aps(self):
        for device in self.devices:
            access_aps_id = []
            for ap in self.aps:
                if device.group == ap.group:
                    access_aps_id.append(ap.id)
                    d_im = math.sqrt(
                        (device.loc[0]-ap.loc[0])**2+(device.loc[1]-ap.loc[1])**2)
                    trans_rate = ap.total_b * math.log(
                        1 + (device.f_p * d_im ** (-self.path_loss)) / self.noise)  # transmit rate
                    device.access_aps.append([ap.id, trans_rate, d_im])
            device.access_aps.sort(key=lambda x: x[2], reverse=True)
            device.access_aps = device.access_aps[:NUM_ACCESS_APS]
            device.access_aps.sort(key=lambda x: x[0])
            device.access_aps = [d[:2] for d in device.access_aps]
            # for ap in self.aps:
            #     if device.group == ap.group:
            #         access_aps_id.append(ap.id)
            # access_aps_id = np.random.choice(access_aps_id, NUM_ACCESS_APS, replace=False)
            # for ap_id in access_aps_id:
            #     ap = self.aps[ap_id]
            #     d_im = math.sqrt(
            #         (device.loc[0]-ap.loc[0])**2+(device.loc[1]-ap.loc[1])**2)
            #     trans_rate = ap.total_b * math.log(
            #         1 + (device.f_p * d_im ** (-self.path_loss)) / self.noise)  # transmit rate
            #     device.access_aps.append([ap.id, trans_rate])
            # device.access_aps.sort(key=lambda x: x[0])
            
            

    def update_ap_offload_info(self, ap_id, device_id):
        self.aps[ap_id].devices_in_ap.append(
            self.devices[device_id])  # 选择卸载到某AP的设备号
        self.aps[ap_id].tasks_input_in_ap.append(
            self.devices[device_id].task[0])  # 选择卸载到某AP的任务inputsize
        self.aps[ap_id].tasks_cpu_in_ap.append(
            self.devices[device_id].task[1])  # 选择卸载到某AP的任务cpucycles

    def get_path_between_aps(self):
        # shortest_path
        path_between_aps = [[[] for _ in range(self.M)] for _ in range(self.M)]
        G = nx.from_numpy_matrix(np.array(self.ap_matrix))
        for m in range(self.M):
            for n in range(self.M):
                path_between_aps[m][n] = nx.shortest_path(
                    G, source=m, target=n)

        # for i in range(self.M):
        #     for j in range(self.M):
        #         if i == j:
        #             shortest_path[i][j] = 0
        #         if i != j and shortest_path[i][j] == 0:
        #             shortest_path[i][j] = 99999
        #         if i != j and shortest_path[i][j] == 1:
        #             self.path_between_aps[i][j].append(i)
        #             self.path_between_aps[i][j].append(j)

        # for k in range(self.M):
        #     for i in range(self.M):
        #         for j in range(self.M):
        #             if shortest_path[i][j] > shortest_path[i][k] + shortest_path[k][j]:
        #                 shortest_path[i][j] = shortest_path[i][k] + \
        #                     shortest_path[k][j]
        #                 self.path_between_aps[i][j] = self.path_between_aps[i][k] + \
        #                     self.path_between_aps[k][j]
        # for m in range(self.M):
        #     for n in range(self.M):
        #         self.path_between_aps[m][n] = list(
        #             set(self.path_between_aps[m][n]))

        return path_between_aps

    def get_mds_in_path_between_aps(self, devices_dispatched):
        mds_in_path_between_aps = [[[]
                                    for i in range(self.M)]for j in range(self.M)]
        for m in range(self.M):
            for n in range(self.M):
                for i in devices_dispatched[m][n]:
                    for m_ in range(len(self.path_between_aps[m][n])-1):
                        if m_ != len(self.path_between_aps[m][n])-1:
                            mds_in_path_between_aps[self.path_between_aps[m][n]
                                                    [m_]][self.path_between_aps[m][n][m_+1]].append(i)

        return mds_in_path_between_aps

    def compute_trans_delay(self, device: Device, ap: AP):
        trans_rate = device.get_trans_rate(ap.id)
        wireless_weight = math.sqrt(
            device.task[0] / trans_rate) / ap.get_bandwidth_divide_rule()
        actual_rate = wireless_weight * trans_rate
        trans_delay = device.task[0] / actual_rate
        # print(trans_delay)

        return trans_delay

    def compute_process_delay(self, device: Device = None, ap: AP = None, is_ap=False, divide_rule=None):
        if not is_ap and not device:
            raise ImportError
        if is_ap and not ap:
            raise ImportError

        if not divide_rule and is_ap:
            divide_rule = ap.get_cpu_divide_rule()

        if not is_ap:
            process_delay = device.task[1] / device.f_c
        else:
            cpu_weight = math.sqrt(
                device.task[1] / ap.total_f) / divide_rule
            process_delay = device.task[1] / (ap.total_f * cpu_weight)

        return process_delay

    def compute_dispatch_delay(self, devices_dispatched):
        wired_divide_rule = [[0]*self.M for i in range(self.M)]
        wired_weight = [
            [[0]*self.M for m in range(self.M)] for i in range(self.N)]
        mds_in_path_between_APs = self.get_mds_in_path_between_aps(
            devices_dispatched)
        dispatch_time = [0 for _ in range(self.N)]
        for m in range(self.M):
            for n in range(self.M):
                if m != n:
                    for i in mds_in_path_between_APs[m][n]:
                        wired_divide_rule[m][n] = wired_divide_rule[m][n] + \
                            math.sqrt(
                                self.devices[i].task[0] / self.wired_width[m][n])
        for m in range(self.M):
            for n in range(self.M):
                if m != n:
                    for i in devices_dispatched[m][n]:
                        d_time_tmp = 0
                        for m_ in range(len(self.path_between_aps[m][n])-1):
                            # /wired_divide_rule[self.path_between_aps[m][n][m_]][self.path_between_aps[m][n][m_+1]]
                            wired_weight[i][self.path_between_aps[m][n][m_]][self.path_between_aps[m][n][m_+1]] = math.sqrt(
                                self.devices[i].task[0]/self.wired_width[self.path_between_aps[m][n][m_]][self.path_between_aps[m][n][m_+1]])
                            # print(" wired_weight[i-1][m][n]", wired_weight[i-1][m][n])#------------------
                            d_time_tmp += self.devices[i].task[0]/(wired_weight[i][self.path_between_aps[m][n][m_]][self.path_between_aps[m]
                                                                                                                    [n][m_+1]]*self.wired_width[self.path_between_aps[m][n][m_]][self.path_between_aps[m][n][m_+1]])
                        dispatch_time[i] = d_time_tmp

        return dispatch_time

    def compute_delay(self, device, off_decision, dispatch_decision, dispatch_time):
        trans_delay = 0
        dispatch_delay = 0
        process_delay = 0
        total_delay = 0
        if off_decision == 0:
            process_delay = self.compute_process_delay(
                device=device, is_ap=False)
        else:
            trans_delay = self.compute_trans_delay(
                device, self.aps[off_decision-1])
            dispatch_delay = dispatch_time[device.id]
            process_delay = self.compute_process_delay(
                device, self.aps[dispatch_decision-1], is_ap=True)

        total_delay = trans_delay + dispatch_delay + process_delay

        return trans_delay, dispatch_delay, process_delay, total_delay
