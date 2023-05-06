import numpy as np
import math

#transmission delay
def s_delay(N,M,devices,APs,tasks,devices_in_AP,trans_rate,trans_time,wireless_weight,actual_rate):  #O_profile是offloading decisions,D_profile是dispatching decisions
    bandwith_divide_rule=[0]*M
    for m in range(M):
        #print("devices_in_AP[",m,"]",devices_in_AP[m])
        for i in devices_in_AP[m]:
            #print("trans_rate[",i,"][",m,"]:",trans_rate[i-M][m])
            #print("bandwith_divide_rule[",m,"]:",bandwith_divide_rule[m])
            bandwith_divide_rule[m]=bandwith_divide_rule[m] + math.sqrt(tasks[i-M][1]/trans_rate[i-M][m])
            #print("bandwith_divide_rule[",m,"]:",bandwith_divide_rule[m])
        for i in devices_in_AP[m]:
            wireless_weight[i-M][m]=math.sqrt(tasks[i-M][1]/trans_rate[i-M][m])/bandwith_divide_rule[m]
            actual_rate[i-M][m]=wireless_weight[i-M][m]*trans_rate[i-M][m]
            trans_time[i-M][m]=tasks[i-M][1]/actual_rate[i-M][m]    
    #print("trans_time:",trans_time) 

#dispatching delay---core--network----资源竞争式
def d_delay(N,M,tasks,devices_dispatched,dispatch_time,dispatch_road_width,wired_weight,path_between_APs,MDs_in_path_between_APs):
    #因为变拓扑结构了，方法参数里加一个路径存储的变量path_between_APs还有存储每条路径有那些MD的变量MDs_in_path_between_APs
    wired_divide_rule=[[0]*M for i in range(M)]
    wired_weight=[[[0]*M for m in range(M)] for i in range(N)]
    for m in range(M):
        for n in range(M):
            if m!=n:
                for i in MDs_in_path_between_APs[m][n]:
                        wired_divide_rule[m][n]=wired_divide_rule[m][n]+math.sqrt(tasks[i-M][1]/dispatch_road_width[m][n])
    for m in range(M):
        for n in range(M):
            if m!=n:
                for i in devices_dispatched[m][n]:
                    d_time_tmp=0
                    for m_ in range(len(path_between_APs[m][n])-2):                
                        wired_weight[i-M][path_between_APs[m][n][m_]][path_between_APs[m][n][m_+1]]=math.sqrt(tasks[i-M][1]/dispatch_road_width[path_between_APs[m][n][m_]][path_between_APs[m][n][m_+1]])#/wired_divide_rule[path_between_APs[m][n][m_]][path_between_APs[m][n][m_+1]]
                        #print(" wired_weight[i-1][m][n]", wired_weight[i-1][m][n])#------------------
                        d_time_tmp+=tasks[i-M][1]/(wired_weight[i-M][path_between_APs[m][n][m_]][path_between_APs[m][n][m_+1]]*dispatch_road_width[path_between_APs[m][n][m_]][path_between_APs[m][n][m_+1]])
                    dispatch_time[i-M]=d_time_tmp
   
'''def d_delay(i,M,tasks,devices_dispatched,dispatch_time,dispatch_rate):
    for m in range(M):
        for n in range(M):
            if i in devices_dispatched[m][n] and m!=n:
                #print("dispatch_rate[",m,"][",n,"]",dispatch_rate[m][n])
                #print("tasks[",i,"][1]",tasks[i-M][1])
                dispatch_time[i-M]=tasks[i-M][1]/dispatch_rate[m][n]
    #print("dispatch_time",i,":",dispatch_time[i-M])'''          


#process delay
def c_delay(M,N,APs,devices,tasks,devices_dispatched,process_time,CPU_weight):
    CPU_divide_rule=[0]*M
    local_device=[]#存储没有卸载任务的人
    AP_device=[]#存储卸载的人
    
    for i in range(N):
        local_device.append(i+M)
    for m in range(M):
        #print("devices_dispatched[",m,"]",devices_dispatched[m])
        for n in range(M):
            for i in devices_dispatched[m][n]:
                AP_device.append(i)
                CPU_divide_rule[n]=CPU_divide_rule[n]+math.sqrt(tasks[i-M][2]/APs[n][4])
    for m in range(M):
        for n in range(M):
            for i in devices_dispatched[m][n]:    
                CPU_weight[i-M][m][n]=math.sqrt(tasks[i-M][2]/APs[n][4])/CPU_divide_rule[n]
                process_time[i-M]=tasks[i-M][2]/(APs[n][4]*CPU_weight[i-M][m][n])
    #print("AP_device",AP_device)
    #print("local_device",local_device)
    for i in AP_device:
        #print(i)
        local_device.remove(i)
    for i in local_device:
        process_time[i-M]=tasks[i-M][2]/devices[i-M][3]
    #print("process_time:",process_time)