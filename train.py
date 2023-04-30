from model import *
from method import *
from config import *

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def set_agent(env, method_name, config):
    # --------- 方法配置 ---------- #
    if method_name == 'WoLF':
        for device in env.devices:
            agent = WoLFPHC(env.M, device.access_aps, theta=config['theta'],
                            s_delta_win=config['s_delta_win'], s_delta_loss=config['s_delta_loss'])
            device.link_agent(agent)
        dispatch_agent = DispatchAgent(M=env.M, env=env, is_dispatch=True)
    elif method_name == 'Q-value':
        for device in env.devices:
            agent = QLearningAgent(env.M, device.access_aps, lamda=config['lamda'], theta=config['theta'],
                                   mode='mix')
            device.link_agent(agent)
        dispatch_agent = DispatchAgent(M=env.M, env=env, is_dispatch=True)
    elif method_name == 'DO':
        for device in env.devices:
            agent = WoLFPHC(env.M, device.access_aps, theta=config['theta'],
                            s_delta_win=config['s_delta_win'], s_delta_loss=config['s_delta_loss'])
            # agent = QLearningAgent(env.M, device.access_aps, lamda=config['lamda'], theta=config['theta'],
            #                        mode='mix')
            device.link_agent(agent)
        dispatch_agent = DispatchAgent(M=env.M, env=env, is_dispatch=False)
    elif method_name == 'RO':
        for device in env.devices:
            agent = DeviceAgent(env.M, device.access_aps)
            device.link_agent(agent)
        dispatch_agent = DispatchAgent(M=env.M, env=env, is_dispatch=False)
    elif method_name == 'GO':
        for device in env.devices:
            agent = QLearningAgent(env.M, device.access_aps, lamda=config['lamda'], theta=config['theta'],
                                   mode='mix')
            device.link_agent(agent)
        dispatch_agent = DispatchAgent(M=env.M, env=env, is_dispatch=False)
    elif method_name == 'WoLF_Con':  # 收敛性实验
        for device in env.devices:
            agent = WoLFPHC(env.M, device.access_aps, lamda=config['lamda'], theta=config['theta'],
                            s_delta_win=config['s_delta_win'], s_delta_loss=config['s_delta_loss'])
            device.link_agent(agent)
        dispatch_agent = DispatchAgent(M=env.M, env=env, is_dispatch=False)
    elif method_name == 'Q_Con':
        for device in env.devices:
            agent = QLearningAgent(env.M, device.access_aps, lamda=config['lamda'], theta=config['theta'],
                                   mode='mix')
            device.link_agent(agent)
        dispatch_agent = DispatchAgent(M=env.M, env=env, is_dispatch=False)
    else:
        raise ImportError

    return dispatch_agent


def train(env, method_name, config):
    dispatch_agent = set_agent(env, method_name, config)
    # --------- 训练过程 ---------- #
    avg_total_delay_list = []
    sum_total_delay_list = []
    conver_epoch = 0
    pbar = tqdm(range(config['max_epoches']))
    for epoch in pbar:
        con_flag = True    # 收敛标志用
        sum_total_delay = 0
        # get actions from devices
        offload_decisions = []
        for device in env.devices:
            offload_decision = device.agent.take_action()
            offload_decisions.append(offload_decision)
            if offload_decision != 0:
                env.update_ap_offload_info(offload_decision-1, device.id)

        # get dispatched_decision
        devices_dispatched = [[[] for i in range(env.M)] for i in range(env.M)]
        for m in range(env.M):
            for n in range(env.M):
                devices_dispatched[m][n].clear()
            devices_dispatched[m][m] = copy.deepcopy(env.aps[m].devices_in_ap)
        for m in range(env.M):
            devices_dispatched[m][m] = [d.id for d in devices_dispatched[m][m]]
            env.aps[m].devices_in_ap_dispatched = copy.deepcopy(
                env.aps[m].devices_in_ap)
        dispatch_time = env.compute_dispatch_delay(
            devices_dispatched=devices_dispatched)

        if method_name != 'RO':
            for device in env.devices:
                offload_decision = offload_decisions[device.id]
                dispatch_decision = offload_decisions[device.id]
                trans_delay, dispatch_delay, process_delay, total_delay = env.compute_delay(
                    device=device, off_decision=offload_decision, dispatch_decision=dispatch_decision, dispatch_time=dispatch_time)
                device.update_delay(trans_delay, dispatch_delay, process_delay)

        devices_dispatched, dispatch_decision = dispatch_agent.take_action(
            devices_dispatched=devices_dispatched)
        # print(devices_dispatched)
        for m in range(env.M):
            env.aps[m].devices_in_ap_dispatched.clear()
        for i, d in enumerate(dispatch_decision):
            if d > 0:
                env.aps[d-1].devices_in_ap_dispatched.append(env.devices[i])

        dispatch_time = env.compute_dispatch_delay(
            devices_dispatched=devices_dispatched)

        for device in env.devices:
            # if method_name == 'WoLF' or method_name == 'DO':
            #     old_policy = copy.deepcopy(device.agent.avg_policy)
            offload_decision = offload_decisions[device.id]
            dis_decision = dispatch_decision[device.id]
            trans_delay, dispatch_delay, process_delay, total_delay = env.compute_delay(
                device=device, off_decision=offload_decision, dispatch_decision=dis_decision, dispatch_time=dispatch_time)
            if method_name != 'RO':
                device.update_delay(trans_delay, dispatch_delay, process_delay)
                device.agent.update_policy(offload_decision, total_delay)
                if method_name == 'WoLF' or method_name == 'DO':
                    device.agent.update_avg_policy()
                
                # for p in range(env.M+1):
                #     sum_diff += abs(
                #         device.agent.policy[p]-old_policy[p])
                #     if sum_diff < config['conver_diff']:
                #         con_flag = True and con_flag
                #     else:
                #         con_flag = False

            sum_total_delay += total_delay
            # if device.id == 30:
            #     # print(device.agent.Q)
            #     print(device.agent.policy)
            #     # print(device.agent.avg_policy)
            #     print("epoch %d: device %d, offload: %d, dispatch:%d, trans_delay: %.4f, dispatch_delay: %.8f, process_delay: %.4f, total_delay: %.4f" %
            #           (epoch, device.id, offload_decision, dis_decision, trans_delay, dispatch_delay, process_delay, total_delay))
        avg_total_delay = sum_total_delay / env.N
        avg_total_delay_list.append(avg_total_delay)
        sum_total_delay_list.append(sum_total_delay)
        pbar.set_description("%s | Num_of_AP %d | Epoch %d" %
                             (method_name, env.M, epoch))
        pbar.set_postfix(sum_total_delay=sum_total_delay,
                         avg_total_delay=avg_total_delay)

        if method_name != 'RO':
            # if con_flag:
            #     conver_epoch += 1
            # else:
            #     conver_epoch = 0
            # if conver_epoch == config['conver_epoch']:
            #     break
            if epoch > config['conver_epoch']:
                for av in avg_total_delay_list[-config['conver_epoch']-1: -1]:
                    if abs(av-avg_total_delay) <= config['conver_diff']:
                        con_flag = True and con_flag
                    else:
                        con_flag = False
                if con_flag:
                    break
        

        for ap in env.aps:
            ap.reset()
    # --------- 训练过程 ---------- #

    if method_name == 'RO':
        result = np.average(sum_total_delay_list)
        print('RO 结果: %.4f' % (np.average(sum_total_delay_list)))
        return result
    result = np.average(sum_total_delay_list[-config['conver_epoch']:])
    print('%s 结果: %.4f' % (method_name, result))

    # if epoch != max_epoches-1:
    #     result = np.average(sum_total_delay_list[-10:])
    #     print('%s 结果: %.4f' % (method_name, epoch, result))
    # else:
    #     result = np.average(sum_total_delay_list[-10:])
    #     print('%s 结果: %.4f' % (method_name, np.average(sum_total_delay_list[-10:])))

    return result

    # ---------- 作图 ------------- #
    # print(sum_total_delay_list)
    # plt.ylabel('sum_total_delay_list')
    # plt.plot(range(len(sum_total_delay_list)), sum_total_delay_list)
    # plt.show()
    # ---------- 作图 ------------- #
