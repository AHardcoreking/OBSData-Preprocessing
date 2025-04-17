#########为了方便修改参数，调整等值线图 单独剥离出来
#########数据读取 /Users/longking/Desktop/ppyy/等值线RMS数据
#########到时差方法 对实际记录数据反演 绘等值线图





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
import ast
import matplotlib as mpl


# fontlabel3=fm.FontProperties(family='Kai',size=10)
# fontlabel2=fm.FontProperties(family='Times New Roman',size=18)
plt.rcParams['font.family']=['SimHei']
mpl.rcParams['axes.unicode_minus'] = False    #显示负号
fontlabel1=fm.FontProperties(family='SimHei',size=7.5)
fd= {'family': 'SimHei', 'size': 7.5}

OBS_num=27
vmean=1555

filepath=f'/Users/longking/Desktop/ppyy/WQ{OBS_num}到时差方法{vmean}/'

x_grid = np.loadtxt(f'/Users/longking/Desktop/ppyy/实测到时差等值线RMS数据/WQ{OBS_num}到时差方法{vmean}/xgrid.txt', dtype=float)
y_grid = np.loadtxt(f'/Users/longking/Desktop/ppyy/实测到时差等值线RMS数据/WQ{OBS_num}到时差方法{vmean}/ygrid.txt', dtype=float)
z_grid = np.loadtxt(f'/Users/longking/Desktop/ppyy/实测到时差等值线RMS数据/WQ{OBS_num}到时差方法{vmean}/zgrid.txt', dtype=float)

with open(f'/Users/longking/Desktop/ppyy/实测到时差等值线RMS数据/WQ{OBS_num}到时差方法{vmean}/parameter.txt', 'r') as file:
    lines = file.readlines()

# 去掉每行末尾的换行符
parameter = [line.strip() for line in lines]

print(lines)

initialcenter=ast.literal_eval(parameter[0])
correctedcenter=ast.literal_eval(parameter[1])
vmean=int(parameter[2])
OBS_num=parameter[3]
lminrms=float(parameter[4])
lmaxrms=float(parameter[5])
xmin=float(parameter[6])
xmax=float(parameter[7])
ymin=float(parameter[8])
ymax=float(parameter[9])
# print(type(initialcenter))
# print(type(correctedcenter))
# print(type(real))




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
or_diff_file=f"/Users/longking/Desktop/ppyy/landerRelc/OBS-WQ{OBS_num}.sxytdiff"
or_diff_data = np.loadtxt(or_diff_file)
sx=or_diff_data[:,0]-read_center[0]
sy=or_diff_data[:,1]-read_center[1]
sz=10.0







   
deltrms=lmaxrms-lminrms
# ccmm=np.linspace(lminrms,lminrms+deltrms*0.01,7) # 5改成10
# ccm0=ccmm[[0, 1, 2]]


# ccm0=np.linspace(lminrms+deltrms*0.01,0.5,10)
# ccm0=np.unique(np.concatenate((ccmm,ccm0)))


# # sign1=np.array([3.0,9.0])
# sign2=np.arange(520,lmaxrms,20)
# # sign=np.unique(np.concatenate((sign1,sign2)))
# ccm0=np.linspace(lminrms,510,6) 
# # ccm1=np.arange(0.6,1.0,0.1)
# # ccm1=np.arange(0.5,3.0,0.1)
# ccm2=np.linspace(510,lmaxrms,15)
# ccm=np.unique(np.concatenate((ccm0,ccm2)))

# # cmlast=np.setdiff1d(ccm, sign)
# tolerance = 1e-4  # 设置阈值
# cmlast = ccm[~np.isclose(ccm[:, None], sign2, atol=tolerance).any(axis=1)]



sign=np.arange(0.2,1.0,0.1)

ccmm=np.linspace(lminrms,0.17,3)
ccm1=np.linspace(0.17,0.2,4)
ccm00=np.linspace(0.2,0.3,5)
# ccm01=np.linspace(0.2,0.3,6)
# ccm00=np.linspace(0.30,0.4,4)

# ccm1=np.linspace(0.31,0.42,6)
ccm2=np.linspace(0.3,lmaxrms,12)
ccm=np.unique(np.concatenate((ccmm,ccm1,ccm00,ccm2)))


# cmlast=np.setdiff1d(ccm, sign)
tolerance = 1e-4  # 设置阈值
# cmlast = ccm[~np.isclose(ccm[:, None], sign, atol=tolerance).any(axis=1)]
cmlast = ccm[~np.isclose(ccm[:, None], sign).any(axis=1)]




plt.figure(figsize=(3.8, 3.3))
plt.tick_params(axis='both', which='major', labelsize=7.5)  # 设置主刻度的字体大小
plt.tick_params(axis='both', which='minor', labelsize=7.5)  # 设置次刻度的字体大小（可选）
contour=plt.contourf(x_grid, y_grid, z_grid,ccm,cmap="jet")
# plt.colorbar(ticks=np.arange(4,38,2))
cbar=plt.colorbar()
cbar.set_label('RMS/ms', fontdict=fd)
cbar.ax.tick_params(labelsize=7.5)
# C_black = plt.contour(x_grid, y_grid, z_grid, levels=np.arange(np.min(ccm), np.max(ccm) + 0.5, 0.5), colors='black',linewidths=0.2)
C_white = plt.contour(x_grid,y_grid,z_grid,cmlast,colors='white',linewidths=0.2)    #绘制所有等值线
C_lable = plt.contour(x_grid,y_grid,z_grid,sign,colors="black",linewidths=0.4)   #标出0.5的整数倍的等值线
plt.clabel(C_lable,inline=True,fmt="%1.1f",fontsize=7.5)



valid_points = np.logical_and.reduce([xmin<= sx, sx <= xmax, ymin <= sy, sy <= ymax])
plt.scatter(sx[valid_points], sy[valid_points], marker='o', color='black', s=6,label='气枪炮点')  

plt.scatter(*initialcenter, marker='x', color='purple', label='投放点位置',s=10,linewidths=0.8)
plt.scatter(*correctedcenter, marker='+', color='red', label='校正结果位置',s=10,linewidths=0.8)




# draw the shots lines





plt.xlabel("东/m",fontproperties=fontlabel1,labelpad=3)
plt.ylabel("北/m",fontproperties=fontlabel1,labelpad=2)
# print(sx[valid_points])
# print(sy[valid_points])
plt.legend(prop=fontlabel1)

# plt.title(f'WQ{OBS_num} Contour Plot of RMS')
plt.savefig(f'/Users/longking/Desktop/ppyy/landerRelc/实测WQ{OBS_num}到时差_{vmean}/ALLRMS_Vary.pdf',dpi=300)
plt.savefig(f'/Users/longking/Desktop/ppyy/landerRelc/实测WQ{OBS_num}到时差_{vmean}/ALLRMS_Vary.svg',dpi=300)
# time.sleep(3)
plt.show()

