###########采用基于到时差的校正方法，采用实际观测数据，使用平均速度速度进行重定位
############使用数据 ppyy 旅行时正演newppyy
############输出路径 ppyy 实测数据到时差反演.txt
############炮号间隔为1
########最后的标记 去除、拆分本应该在一个方向上的炮点为sxy1 sxy2

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from scipy.interpolate import interp2d,griddata
import re
import os
import time
from matplotlib import cm
from matplotlib import colors
import h5py
import matplotlib.font_manager as fm

fontlabel1=fm.FontProperties(family='Kai',size=10)
fontlabel2=fm.FontProperties(family='Times New Roman',size=18)

plt.rcParams['font.family']=['Kai']
plt.rcParams['axes.unicode_minus'] = False    #显示负号


def add_noise(signal,mean,std_dev): #添加背景噪声函数
    noise=np.random.normal(mean,std_dev,size=len(signal))
    noisy_signal=signal+noise
    return noisy_signal


def get_Tt(x, y, Spoints):      #实现对正演旅行时的双线性插值函数
            rx = math.ceil(x)
            ry = math.ceil(y)
            a= [rx-1,ry-1,Spoints[rx-1,ry-1]]
            b= [rx-1,ry,Spoints[rx-1,ry]]
            c= [rx,ry-1,Spoints[rx,ry-1]]
            d= [rx,ry,Spoints[rx,ry]]
            
            return(
               a[2] * (rx-x) * (ry-y) +
               b[2] * (x-rx+1) * (ry-y) +
               c[2] * (rx-x) * (y-ry+1) +
               d[2] * (x-rx+1) * (y-ry+1)
            )
     

OBS_num=3
vmean=1555
# 随机选取一个实际坐底点，计算理论到时
np.set_printoptions(threshold=np.inf)    #设置数组print时可以完整打印出来，不会出现省略号...

def generate_unique_points(center, radius, num_points):       #  生成随机点并避免重复函数
    unique_points = set()
    while len(unique_points) < num_points:
        # 生成半径和角度
        r = np.sqrt(np.random.uniform(0, radius**2, num_points - len(unique_points)))
        theta = np.random.uniform(0, 2 * np.pi, num_points - len(unique_points))

        # 生成坐标
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)

        # 添加到集合中
        new_points = set(zip(x, y))
        unique_points.update(new_points)

    # 转换为数组并截取所需数量的点
    coordinates = np.array(list(unique_points))[:num_points]
    return coordinates

center=(0,0)
radius=1500
num_points=1



# 读取投放点坐标    为了归一化地形和炮点坐标 建议保留
dep_file=f"/Users/longking/Desktop/ppyy/landerRelc/lander.deg.xy"
with open(dep_file, 'r') as file:
    dep = file.read()
pattern = re.compile(fr'^OBS-WQ{OBS_num}\b.*$', re.MULTILINE)
matches = pattern.findall(dep)
for match in matches:
    split_array = match.split()
    x=float(split_array[3])
    y=float(split_array[4])
read_center=list((x,y))

# 读取炮点坐标并进行归一化
or_diff_file=f"/Users/longking/Desktop/ppyy/landerRelc/OBS-WQ{OBS_num}.allsxy"
or_diff_data = np.loadtxt(or_diff_file)
sx=or_diff_data[:,0]-read_center[0]
sy=or_diff_data[:,1]-read_center[1]
sz=10.0
# 读取实测数据到时差
or_diff=or_diff_data[:,2]
Re_diff_data=or_diff[1:]-or_diff[:-1]





# 读取地形深度数据
topo_xyz_file=f"/Users/longking/Desktop/ppyy/landerRelc/OBS-WQ{OBS_num}.xyz.topo.topoxyz"
topo_input_data = np.loadtxt(topo_xyz_file)
topo_input_data[0:,0]= topo_input_data[0:,0]*1-read_center[0]
topo_input_data[0:,1]= topo_input_data[0:,1]*1-read_center[1]
tpx0 = topo_input_data[0:,0]
tpy0 = topo_input_data[0:,1]
tpz0 = topo_input_data[0:,2]*-1
# realz = griddata(topo_input_data[0:,0:2],tpz0,real,method='nearest')        #将生成的随机点（作为实际点）所处深度插值出来


# 读取实际CTD速度数据                                   #用不上 删掉没关系
avt_file=f"/Users/longking/Desktop/newppyy/landerRelc/OBS-WQ3-tarveltime.h5"          
with h5py.File (avt_file,"r") as f:
    dataset = f["traveltime"]    
    Timedata = dataset[:]             #z,x
Timedatatime=Timedata.transpose()     #x,z
print("读取h5OK")

# print(realz[0])






                          
# print(ttt)

# #绘制到时分布图
# folder_path = f'/Users/longking/Desktop/ppyy/landerRelc/WQ3cor_curve/WQ{OBS_num}到时差_{vmean}'
# os.makedirs(folder_path, exist_ok=True)

# OF=np.sqrt((sx-real[0])**2 + (sy-real[1])**2)
# flag=(sx-real[0])/(np.absolute(sx-real[0]))
# OF=OF*flag/1000  #水平偏移距 横坐标km
# # OF=OF[:-1]
# print(OF)
# Tr=ttt
# print(Tr)
# plt.figure(figsize=(14,12))
# plt.scatter(OF,Tr,color='blue',marker='+',s=25)
# plt.xlabel("偏移距/km",fontproperties=fontlabel1,labelpad=10)
# plt.ylabel("相对于第一炮到时差/s",fontproperties=fontlabel1,labelpad=10)
# plt.gca().invert_yaxis()
# plt.grid()
# plt.xticks(fontproperties=fontlabel2)
# plt.yticks(fontproperties=fontlabel2)
# plt.savefig(f'/Users/longking/Desktop/newexamppyy/landerRelc/WQ3cor_curve/WQ{OBS_num}到时差_{vmean}/到时曲线.pdf',dpi=300)
# plt.show()
# print(vmean)
# exit(0)



# 0 输入需要校正的OBS序号以及停止迭代的约束条件
# 输入一组OBS序号
user_input = input("input several OBS_nums: ")
numbers = [str(num) for num in user_input.split()]


minRadius = input("Input MinRadius(m)=")
maxInverTime = input("Input MaxInverTime=")
minRms = input("Input MinRMS=")
num_points = input("InverPointsNum=")
vmean = input("Mean Velocity of Water=")
initialRadius = input("Initial Radius=")
print("Parameters Set has been Completed.")

if minRadius == "":
    minRadius = 20
if maxInverTime == "":
    maxInverTime = 18
if minRms == "":
    minRms = 1e-06 #ms
if initialRadius == "":
    initialRadius = 1500
if num_points == "":
    num_points = 2000
if vmean == "":                 #默认平均声速1540
    vmean = 1555


# 遍历每一台OBS
for n in numbers:

    OBS_num = n
    folder_path = f'/Users/longking/Desktop/ppyy/landerRelc/实测WQ{OBS_num}到时差_{vmean}'
    os.makedirs(folder_path, exist_ok=True)



    # minRadius = 100 m
    # maxInverTime = 100
    # minRms = 1e-05 ms
    # initialRadius = 1500 m
    # num_points = 2000
    # vmean = 1540 m/s
    initialcenter=(0,0)
    center=(0,0)  # Already normalized
    CurrentInverTime = 1
    searchRadius = initialRadius
    searchMinRms = np.zeros((maxInverTime))


    # 1 Read in Deployment Position   重复读投放点坐标
    dep_file=f"/Users/longking/Desktop/ppyy/landerRelc/lander.deg.xy"
    with open(dep_file, 'r') as file:
        dep = file.read()


    pattern = re.compile(fr'^OBS-WQ{OBS_num}\b.*$', re.MULTILINE)
    matches = pattern.findall(dep)
    for match in matches:
        split_array = match.split()
        x=float(split_array[3])
        y=float(split_array[4])
    read_center=(x,y)
    # print((read_center))


    # 2 Read in Topo:  重复读地形深度
    topo_xyz_file=f"/Users/longking/Desktop/ppyy/landerRelc/OBS-WQ{OBS_num}.xyz.topo.topoxyz"
    topo_input_data = np.loadtxt(topo_xyz_file)
    topo_input_data[0:,0]= topo_input_data[0:,0]*1-read_center[0]
    topo_input_data[0:,1]= topo_input_data[0:,1]*1-read_center[1]
    tpx0 = topo_input_data[0:,0]
    tpy0 = topo_input_data[0:,1]
    tpz0 = topo_input_data[0:,2]*-1

    tpx0_len = len(np.unique(tpx0))
    tpy0_len = len(np.unique(tpy0))

    tpx0_grd = tpx0.reshape(tpx0_len,tpy0_len)
    tpy0_grd = tpy0.reshape(tpx0_len,tpy0_len)
    tpz0_grd = tpz0.reshape(tpx0_len,tpy0_len)



    # 3 Read RealTDiff & sx,xy  重复读炮点坐标

    or_diff_file=f"/Users/longking/Desktop/ppyy/landerRelc/OBS-WQ{OBS_num}.allsxy"
    or_diff_data = np.loadtxt(or_diff_file)
    sx=or_diff_data[:,0]-read_center[0]
    sy=or_diff_data[:,1]-read_center[1]
    sz=10.0
    






    fig, ax = plt.subplots()
    fig.set_size_inches(6.4, 4.8)
    fig.set_dpi(300)
    # 4 Get Random Points and Its Topo & Calculate RMS & Iteration 
    while CurrentInverTime <= maxInverTime:
        # Generate random points & its topo
        coordinates = generate_unique_points(center, searchRadius, num_points)
        if  CurrentInverTime==1:
            x0 = coordinates[:,0]
            y0 = coordinates[:,1]
            # z0 = griddata(topo_input_data[0:,0:2],tpz0,(x0,y0),method='nearest')
        else:
            #Judge and remove outliers and then supply adequate random points
            if initialRadius-searchRadius<math.sqrt((center[0]-0)**2+(center[1]-0)**2)<initialRadius+searchRadius:
                # print('out of borders')

                condition=np.sqrt((coordinates[:,0]-0)**2+(coordinates[:,1]-0)**2)<=initialRadius
                select_points=coordinates[condition]

                select_points_x=select_points[:,0]
                select_points_y=select_points[:,1]
                lenmiss = num_points - len(select_points_x)
                # print(lenmiss)

                

                while lenmiss >0:
                    add_points = generate_unique_points(center, searchRadius, lenmiss)
                    condition=np.sqrt((add_points[:,0]-0)**2+(add_points[:,1]-0)**2)<=initialRadius
                    add_filtered=add_points[condition]
                    mask=np.isin(add_filtered, select_points).all(axis=1)
 
                    select_points = np.concatenate((select_points, add_filtered), axis=0)
                    select_points_x=select_points[:,0]
                    lenmiss = num_points - len(select_points_x) 
                    # print(lenmiss)
                coordinates=select_points
                # print(coordinates.shape,'It is OK')
            #print(coordinates.shape,'The Final One')
            x0 = coordinates[:,0]
            y0 = coordinates[:,1]
            # print("The real points now：",len(x0),len(y0))
        z0 = griddata(topo_input_data[0:,0:2],tpz0,(x0,y0),method='nearest')

        # calculate the SimulatedDiff and RMS
        z0_rms = np.zeros((num_points,1))
        
        
        for ni in range(num_points):
            ttt=[]
            # 计算生成随机点对各炮的偏移距和所在深度 
            offset = np.sqrt((sx-x0[ni])**2 + (sy-y0[ni])**2+(sz-z0[ni])**2)
            # print(offset.shape)
            # np.savetxt("/Users/longking/Desktop/offset.txt",offset)
            # print("偏移距最大值",np.max(offset))
            
            ttt=offset/vmean
            ttt=ttt[1:]-ttt[:-1]#使用平均速度
            
            # for m in offset:                                  #使用CTD速度
            #     ttt.append(get_Tt(m,z0[ni],Timedatatime))
            # print("插值结果出来了",ttt)
            arvT=np.array(ttt)
    
            # arvT=add_noise(arvT,0,0.001)
            Sim_diff_data = arvT
            z0_rms[ni] = (np.sum(np.sqrt((Sim_diff_data - Re_diff_data)**2))/num_points)*1000     #*1000  ms

        # get minimum rms & its x, y
        min_index_flat = np.argmin(z0_rms)
        z0_rms_min_index  = np.unravel_index(min_index_flat, z0_rms.shape)
        #print(z0_rms_min_index)

        minx0 = x0[min_index_flat]
        miny0 = y0[min_index_flat]

        #print(CurrentInverTime,z0_rms[z0_rms_min_index])

    
        ax.set_xlim(-initialRadius, initialRadius)
        ax.set_ylim(-initialRadius, initialRadius)	

        if CurrentInverTime==1:
            sumzRms=z0_rms
            sumx0=x0
            sumy0=y0
            bigRMS=math.ceil(np.max(z0_rms))
            searchMinRms[CurrentInverTime] = z0_rms[z0_rms_min_index]
            Last_min_RMS=searchMinRms[CurrentInverTime]
            LastsearchRadius=searchRadius
            Last_center=center
            print(searchMinRms[CurrentInverTime])
            plt.scatter(x0, y0, c=z0_rms, cmap='turbo', vmin=0,vmax=bigRMS,s=0.2)
            plt.colorbar(label='RMS/ms')
            # plt.title(f'WQ{OBS_num} IterationTimes:{CurrentInverTime}')
            plt.xlabel("东/m",fontproperties=fontlabel1,labelpad=6)
            plt.ylabel("北/m",fontproperties=fontlabel1,labelpad=4)
            plt.savefig(f'/Users/longking/Desktop/ppyy/landerRelc/实测WQ{OBS_num}到时差_{vmean}/Time00{CurrentInverTime}.pdf',dpi=300)
            CurrentInverTime += 1

        else:
            if z0_rms[z0_rms_min_index]>=Last_min_RMS:
                searchRadius = searchRadius*0.8 
                # print("bigger RMS, continue")
            else:
                sumzRms= np.concatenate((sumzRms, z0_rms), axis=0)
                sumx0=np.concatenate((sumx0, x0), axis=0)
                sumy0=np.concatenate((sumy0, y0), axis=0)
                searchMinRms[CurrentInverTime] = z0_rms[z0_rms_min_index]
                print(searchMinRms[CurrentInverTime])
                Last_min_RMS=searchMinRms[CurrentInverTime]
                Last_center=center
                plt.scatter(x0, y0, c=z0_rms, cmap='turbo', vmin=0,vmax=np.max(bigRMS),s=0.2)
                # plt.title(f'WQ{OBS_num} IterationTimes:{CurrentInverTime}')
                plt.savefig(f'/Users/longking/Desktop/ppyy/landerRelc/实测WQ{OBS_num}到时差_{vmean}/Time00{CurrentInverTime}.pdf',dpi=300)
                center = (minx0, miny0)
                LastsearchRadius=searchRadius
                searchRadius = searchRadius*0.8
                if searchRadius < minRadius:
                    print("searchRadius < minRadius")
                    break
                dtrms = abs(searchMinRms[CurrentInverTime] - searchMinRms[CurrentInverTime-1])
                if dtrms < minRms:
                    print("dtrms < minRms")
                    break
                CurrentInverTime += 1
        plt.pause(1) 


    plt.show(block=False)
    plt.pause(3)
    plt.close()

    # Show the Final Position Result
    correctedcenter=(minx0,miny0)
    print(correctedcenter)
    # print(searchMinRms)

    # Show the RMS Variation
    nonzero_indices = np.nonzero(searchMinRms)
    # print(nonzero_indices)
    y=searchMinRms[nonzero_indices] 
    x= list(range(1, CurrentInverTime+1))
    # plt.clf()
    fig=plt.figure()
    fig.set_size_inches(6.4, 4.8)
    fig.set_dpi(300)
    plt.plot(x, y, marker='o')
    # plt.title('RMS Varation')
    plt.grid()
    plt.xlabel('迭代次数',fontproperties=fontlabel1,labelpad=6)
    plt.ylabel('最小RMS值/ms',fontproperties=fontlabel1,labelpad=5)
    # plt.xlabel(f'WQ{OBS_num} Iteration Times')
    # plt.ylabel('RMS')
    # plt.legend()
    plt.xticks(x)

    plt.savefig(f'/Users/longking/Desktop/ppyy/landerRelc/实测WQ{OBS_num}到时差_{vmean}/RMS_Vary.pdf',dpi=300)
    plt.show(block=False)
    plt.pause(8)
    plt.close() 
   
   
   
   
   

    x_grid, y_grid = np.meshgrid(np.unique(sumx0), np.unique(sumy0))
    z_grid = griddata((sumx0, sumy0), sumzRms, (x_grid, y_grid), method='linear')
    z_grid = np.squeeze(z_grid)
    
    
    lminrms=np.min(sumzRms)
    lmaxrms=np.max(sumzRms)
    xmin=np.min(sumx0)
    xmax=np.max(sumx0)
    ymin=np.min(sumy0)
    ymax=np.max(sumy0)

    folder_path = f'/Users/longking/Desktop/ppyy/实测到时差等值线RMS数据/WQ{OBS_num}到时差方法{vmean}'
    os.makedirs(folder_path, exist_ok=True)
    np.savetxt(f"/Users/longking/Desktop/ppyy/实测到时差等值线RMS数据/WQ{OBS_num}到时差方法{vmean}/xgrid.txt", x_grid, fmt='%f', delimiter=' ')
    np.savetxt(f"/Users/longking/Desktop/ppyy/实测到时差等值线RMS数据/WQ{OBS_num}到时差方法{vmean}/ygrid.txt", y_grid, fmt='%f', delimiter=' ')
    np.savetxt(f"/Users/longking/Desktop/ppyy/实测到时差等值线RMS数据/WQ{OBS_num}到时差方法{vmean}/zgrid.txt", z_grid, fmt='%f', delimiter=' ')
    parameter=list([initialcenter,correctedcenter,vmean,OBS_num,lminrms,lmaxrms,xmin,xmax,ymin,ymax])
    print(parameter)
    with open(f"/Users/longking/Desktop/ppyy/实测到时差等值线RMS数据/WQ{OBS_num}到时差方法{vmean}/parameter.txt",'w') as file:
        for item in parameter:
            file.write(f"{item}\n")
    

    # lminrms=np.min(sumzRms)
    # lmaxrms=np.max(sumzRms)
   
    # deltrms=lmaxrms-lminrms
    # # ccmm=np.linspace(lminrms,lminrms+deltrms*0.01,7) # 5改成10
    # # ccm0=ccmm[[0, 1, 2]]
    
    
    # # ccm0=np.linspace(lminrms+deltrms*0.01,0.5,10)
    # # ccm0=np.unique(np.concatenate((ccmm,ccm0)))
    
    # sign=np.arange(0.5,3.5,0.25)
    # # sign1=np.array([3.0,9.0])
    # # sign2=np.arange(15,lmaxrms,10)
    # # sign=np.unique(np.concatenate((sign1,sign2)))
    # ccm0=np.linspace(lminrms,0.2,10) 
    # # ccm1=np.arange(0.6,1.0,0.1)
    # ccm1=np.arange(0.2,lmaxrms,0.1)
    # # ccm2=np.linspace(3.0,lmaxrms,15)
    # ccm=np.unique(np.concatenate((ccm0,ccm1)))
    
    # # cmlast=np.setdiff1d(ccm, sign)
    # tolerance = 1e-4  # 设置阈值
    # cmlast = ccm[~np.isclose(ccm[:, None], sign, atol=tolerance).any(axis=1)]
    # contour=plt.contourf(x_grid, y_grid, z_grid,ccm,cmap="jet")
    # # plt.colorbar(ticks=np.arange(4,38,2))
    # plt.colorbar()
    # # C_black = plt.contour(x_grid, y_grid, z_grid, levels=np.arange(np.min(ccm), np.max(ccm) + 0.5, 0.5), colors='black',linewidths=0.2)
    # C_white = plt.contour(x_grid,y_grid,z_grid,cmlast,colors='white',linewidths=0.2)    #绘制所有等值线

    # C_lable = plt.contour(x_grid,y_grid,z_grid,sign,colors="black",linewidths=0.4)   #标出0.5的整数倍的等值线
    # plt.clabel(C_lable,inline=True,fmt="%1.1f",fontsize=8)



 

    # plt.scatter(*initialcenter, marker='+', color='blue', label='Deployment Point',s=10)
    # plt.scatter(*correctedcenter, marker='+', color='red', label='Corrected Point',s=0.05)
    # plt.scatter(*real, marker='+', color='yellow', label='Corrected Point',s=0.05)


    # # draw the shots lines
    # xmin=np.min(sumx0)
    # xmax=np.max(sumx0)
    # ymin=np.min(sumy0)
    # ymax=np.max(sumy0)


   

    # valid_points = np.logical_and.reduce([xmin<= sx, sx <= xmax, ymin <= sy, sy <= ymax])
    # plt.scatter(sx[valid_points], sy[valid_points], marker='o', color='black', s=10,label='Positions of Shots')  
    # # print(sx[valid_points])
    # # print(sy[valid_points])
    # plt.legend()


    # plt.title(f'WQ{OBS_num} Contour Plot of RMS')
    # plt.savefig(f'/Users/longking/Desktop/newexamppyy/landerRelc/WQ{OBS_num}到时差_{vmean}/ALLRMS_Vary.pdf',dpi=300)
    # plt.savefig(f'/Users/longking/Desktop/newexamppyy/landerRelc/WQ{OBS_num}到时差_{vmean}/ALLRMS_Vary.svg',dpi=300)
    # # time.sleep(3)
    # plt.show(block=False)
    # plt.pause(3)
    # # plt.clf()
    # plt.close()

  
    result_pos_path=f"/Users/longking/Desktop/ppyy/landerRelc/实测数据到时差反演_{vmean}.txt"
    finalpos=list([minx0+read_center[0],miny0+read_center[1]])
    fpos=list([minx0,miny0])
    drift_distance = math.sqrt((minx0)**2+(miny0)**2)       #求校正结果相对投放点漂移距离
   
    minrms=np.min(searchMinRms[nonzero_indices] )           #求校正结果所处位置的RMS值
    read_centerz=griddata(topo_input_data[0:,0:2],tpz0,read_center,method='nearest') #将投放点（初始迭代中心）所处深度插值出来
    finalz=griddata(topo_input_data[0:,0:2],tpz0,fpos,method='nearest') #将校正结果点所处深度插值出来
    
    
    with open(result_pos_path, 'a') as file:
        write_data=f'WQ{OBS_num}  '+str(vmean)+'  '+str(read_center[0])+'  '+str(read_center[1])+'  '+str(read_centerz)+'  '+str(minx0+read_center[0])+'  '+str(miny0+read_center[1])+'  '+str(finalz)+'  '+str(drift_distance)+'  '+str(minrms)+"\n"
        file.write(write_data)
print('The position correction is finished')
