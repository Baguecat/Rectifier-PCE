import numpy as np
import matplotlib.pyplot as plt
import cmath
from scipy.integrate import cumtrapz, simps # Import cumtrapz and simps

# Basic constants and parameter definitions
sqrt = np.sqrt
pi = np.pi
e = np.e

# Define constants
k12 = 0.04
Q1 = 133.34
Q2 = 19.2
L1 = 559e-9  # 559 nH
L2 = 63e-9   # 63 nH
f = 40.68e6  # 40.68 MHz
w = 2 * pi * f  # Angular frequency
ωp = w

# Resistor and capacitor calculation
R1 = w * L1 / Q1
R2 = w * L2 / Q2   # R2 calculation
C1 = 1 / (w**2 * L1)
C2 = 1 / (w**2 * L2)
M12 = k12 * sqrt(L1 * L2)


R1 = w * L1 / Q1
R2 = w * L2 / Q2   # R2 calculation
C1 = 1 / (w**2 * L1)
C2 = 1 / (w**2 * L2)
M12 = k12 * sqrt(L1 * L2)

# Input voltage and other parameters
V21_force = 11.54 / 2  # Half of input voltage

RSW = 0.5
force_en = 0

Vs = 0.29
V21 = Vs * w * M12 / (R1 + (w**2 * M12**2) / (R2 + RSW))
Req=R2+RSW



vcross=0.2
RSW_HV=15

#VL_k_1=3.638   #1.57
#N0=40

#VL_k_1=3.515   #1.57
#N0=50

VL_k_1=2.4   #1.57           
Cpar=24e-12*2                
N0=78                        

RHV=100e3    


VLV=0.9672   


#############################   manual change ###############################
VLV_min=0.9647471
N0_lower_manual=40 
##############################################################

iL2_max=80e-3 
CLV=2e-9 
#RLV=100      
RLV=200
'''   RLV=200
N0=2, VLV_MIN=0.7062098
N0=3, VLV_MIN=0.859304
N0=4, VLV_MIN=0.902073
N0=6, VLV_MIN=0.932088
N0=8, VLV_MIN=0.9437051
N0=10, VLV_MIN=0.9522226
N0=12, VLV_MIN=0.95653
N0=15, VLV_MIN=0.9618269
N0=20, VLV_MIN=0.9663228
N0=30, VLV_MIN=0.9649938
N0=40, VLV_MIN=0.9647471
N0=50, VLV_MIN=0.967553
N0=60, VLV_MIN=0.968137
N0=10000, VLV_MIN=1.022
'''

'''   RLV=100

N0=2, VLV_MIN=0.7112
N0=3, VLV_MIN=0.7813
N0=4, VLV_MIN=0.8048
N0=6, VLV_MIN=0.8241
N0=8, VLV_MIN=0.8327
N0=10, VLV_MIN=0.8368
N0=12, VLV_MIN=0.839
N0=15, VLV_MIN=0.8425
N0=20, VLV_MIN=0.8435
N0=30, VLV_MIN=0.8435
N0=40, VLV_MIN=0.848
N0=50, VLV_MIN=0.8423
N0=60, VLV_MIN=0.8467
N0=10000, VLV_MIN=0.9193
														

'''



beta=0.778   #0.778
###############################

VL_k_1=VL_k_1-vcross
CL=200e-12
CHV=CL

tch=2e-9
#t1=Cpar*VL_k_1/(iL2_max/2)

fenzi1=-4*Cpar*VL_k_1**2*(CL-Cpar)
fenzi2=iL2_max*tch*(4*CL*VL_k_1-4*Cpar*VL_k_1+iL2_max*tch )
fenzi=fenzi1+fenzi2
fenmu=2*CL*iL2_max*tch
VL_k=fenzi/fenmu

A=iL2_max/Cpar
B=VL_k_1/A
t1=tch-sqrt(tch**2-2*tch*B)
t2=tch-t1
delta_VHV_linear_tcasi=iL2_max*t1**2/tch/2/CL

omega=1/sqrt(L2*Cpar)
t1_nodamping=1/omega*np.arcsin(VL_k_1*Cpar*omega/iL2_max)
t_val=t1_nodamping # Renamed t to t_val to avoid conflict with time array 't'
t2_nodamping=tch-t1_nodamping
alpha=R2/2/L2
wd=sqrt(1/L2/Cpar-alpha**2)
w_nodamping=sqrt(1/L2/Cpar)
w_nodamping_t2=sqrt(1/L2/(Cpar+CHV))
iL2_t_damping=iL2_max*np.exp(-alpha*t_val)*np.cos(wd*t_val)


iL2_t_nodamping=iL2_max*np.exp(-alpha*t1_nodamping)*np.cos(w_nodamping*t1_nodamping)
iL2_t_nodamping_slope=-iL2_max*np.sin(w_nodamping*t1_nodamping)*w_nodamping
iL2_t2_end=iL2_t_nodamping+iL2_t_nodamping_slope*t2_nodamping      ## good 

#alpha_t2=(R2+RSW_HV*1)/2/L2
alpha_t2=(R2)/2/L2
#alpha_t2=100
wd_t2=1/sqrt(1/L2/CHV-alpha_t2**2)
iL2_t_damping_t2_end=iL2_t_nodamping*np.exp(-alpha_t2*t2_nodamping)*np.cos(wd_t2*t2_nodamping)
#iL2_t_damping_t2_end=iL2_max*np.exp(-alpha*tch)*np.cos(w_nodamping*tch)
iL2_t_damping_t2_end=iL2_max*np.exp(-alpha_t2*tch)*np.cos(wd_t2*tch)
iL2_t_damping_t2_end=iL2_max*np.exp(-alpha_t2*t2_nodamping)*np.cos(wd_t2*t2_nodamping)


iL2_t_linear=iL2_max-iL2_max/tch*t1
a1=tch*np.sin(w_nodamping_t2*tch)/w_nodamping_t2-np.cos(w_nodamping_t2*tch)/(w_nodamping_t2)**2
a2=t1_nodamping*np.sin(w_nodamping_t2*t1_nodamping)/w_nodamping_t2-np.cos(w_nodamping_t2*t1_nodamping)/(w_nodamping_t2)**2
delta_VHV_nodamping=-iL2_max/CHV*((np.sin(w_nodamping_t2*tch)-np.sin(w_nodamping_t2*t1_nodamping))/w_nodamping_t2 -1/tch*(    (a1) - (a2)     )    )

#iL2_end=iL2_t_nodamping-iL2_t_nodamping/tch*t2_nodamping



#####################################################################################################################################################
delta_VHV_linear_my=(iL2_t_nodamping+iL2_t2_end)/2*t2_nodamping/CL
#####################################################################################################################################################


VL_k=VL_k_1+delta_VHV_linear_my


#####################################################################################################################################################
VHV_dec=VL_k-VL_k*np.exp(-N0/f/RHV/CHV)
#####################################################################################################################################################


VLV_dec=VLV-VLV*np.exp(-2/f/RLV/CLV)
Rmos=0.5
#Vripple=(VLV/RLV)/f/CLV*(1-np.exp(-f/Rmos/CLV))
Vripple=(VLV/RLV)/f/CLV*(1-np.exp(-f/RLV/CLV))
Vripple_approximation=(VLV/RLV)/2/f/CLV

Vcharging_LV=VLV*(1-np.exp(-1/f/2/RLV/CLV))

#####################################################################################################################################################
Vripple_fall=(VLV/RLV)/2/f/CLV*beta
Vripple_rise=(VLV/RLV)/2/f/CLV*(1-beta)
#####################################################################################################################################################

#####################################################################################################################################################
Vripple_VLV=Vripple_fall*4+Vripple_rise*3
#####################################################################################################################################################


tlv=467e-9
alpha_lv=1/RLV/CLV
wd_lv=sqrt(1/L2/C2-alpha_lv**2)

##################################
#VLV_min=VLV-Vripple_VLV*1

#############

#VLV_tlv=VLV-Vripple_VLV*e**(-tlv/RLV/CLV)*np.cos(wd_lv*tlv)

#I0=39e-3
#ton=3.4133e-9
#term=e**(-alpha_lv*ton)*(alpha_lv*np.cos(wd_lv*ton)+wd_lv*np.sin(wd_lv*ton))-alpha_lv
#delta_VLV=I0/CLV*term/(alpha_lv**2+wd_lv**2)


vdrop_lv=40e-3
cpar_coeff=0.8

vlv_n1=VLV_min
term1=(vlv_n1-vcross+vdrop_lv*1)*w*C2/(iL2_max*cpar_coeff)
tch_lv=np.arcsin(term1)/w
Il2_tchlv=iL2_max*np.cos(w*tch_lv)


T=1/f
t2_lv=T/4
t1_lv=tch_lv
dT_lv=t2_lv-t1_lv
Il2_t1_lv=iL2_max*np.cos(w*t1_lv)
Il2_t2_lv=iL2_max*np.cos(w*t2_lv)



vlv_ch=iL2_max/w/CLV*( np.sin(w*t2_lv)- np.sin(w*t1_lv)) 
vlv_t1=VLV_min
vlv_t2=vlv_t1+vlv_ch

tdisch_lv=t1_lv+T/4
vlv_disch=vlv_t2*(1-e**(-tdisch_lv/RLV/CLV))
dvlv=vlv_ch-vlv_disch
vlv_t3=vlv_t1+dvlv
t3_lv=T/2+t1_lv


VLV_min_new=VLV_min+dvlv







RSW=2.21


Vs = 0.29
V21 = Vs * w * M12 / (R1 + (w**2 * M12**2) / (R2 + RSW))

Req=R2+RSW
Q2eq=w*L2/Req
#Q2eq=Q2
V21_q=Vs*k12*sqrt(Q1*Q2eq)*sqrt(Req)/(1+k12**2*Q1*Q2eq)/sqrt(R1)   
V21=V21_q
# Define a
#a = R2 / (2 * L2)  # Calculate a
a = (R2+RSW) / (2 * L2)  # Calculate a

# Define time range, modified from 0 to 1 microsecond
t = np.linspace(0, int(N0_lower_manual * 1) / f, 1000)  # From 0 to 1 microsecond

# Define IL2
#IL2 = (abs(V21) * #       ( (-e**(-t * (a + w)) * (-1 + e**(2 * t * w)))*1 + 2 * np.sin(w * t)) ) / (2 * (Req))

Req=R2+RSW
#est=cmath.sqrt(a**2-w**2)


#test=sqrt(w**2-a**2)
#A=-(e**(-t)*(a+test)*(-1+e**(2*t*test))*w)/test
#bracket=A + 2 * np.sin(w * t)
#IL2 = abs(V21) * bracket / (2 * (Req))


#wd=cmath.sqrt(a**2-w**2)
wd=sqrt(w**2-a**2)

bracket=-w*abs(V21)/Req/wd
IL2 = abs(V21)/Req *np.sin(w*t)+ e**(-a*t)*np.sin(wd*t)*bracket*1
#IL2 = abs(V21)/Req *np.sin(w*t)  #stable state
print('IL2_MAX=',max(IL2))
  

# -------------------- Extract maximum values of the positive half-axis (first N0) --------------------
from scipy.signal import find_peaks

# Calculate the length of one period (in seconds)
T = 1 / f

# Get t and IL2
dt = t[1] - t[0]  # Time step
samples_per_half_cycle = int(T / 2 / dt)  # Number of samples per positive half cycle

samples_per_half_cycle = int(T / 2 / dt)  # Number of samples per positive half cycle

# Find all peak indices
peaks, _ = find_peaks(-IL2, height=1e-3)

# Force add the last point (if it's a local peak)
if len(IL2) > 1 and IL2[-2] < IL2[-1]:
    peaks = np.append(peaks, len(IL2) - 1)

#positive_half_indices = [p for p in peaks if np.sin(w * t[p]) > 0]
#positive_half_indices = [p for p in peaks]
positive_half_indices = list(peaks)

# Sort positive half-axis peaks by time, extract the first N0
positive_half_peaks = sorted(positive_half_indices[:N0_lower_manual])
il2_positive_max_values = [-IL2[p] for p in positive_half_peaks]
il2_positive_peaks=il2_positive_max_values
# Output
print(f'\nFirst {N0} positive half-axis IL2(t) peaks:')
for i, val in enumerate(il2_positive_max_values):
    print(f'Cycle {i+1}: IL2 max = {val:.6f} A')














N0_lower=10000
# Calculate VLV_min_new, ensuring it iterates according to rules with N0_lower
VLV_min_new = VLV_min  # Initialize VLV_min_new
num_iterations = int(N0_lower * 1)  # Calculate number of iterations, increasing by 0.5 each time

for _ in range(num_iterations):
    vlv_n1 = VLV_min_new  # Update vlv_n1 to current VLV_min_new
    term1 = (vlv_n1 - vcross + vdrop_lv * 1) * w * C2 / (iL2_max * cpar_coeff)
    tch_lv = np.arcsin(term1) / w
    Il2_tchlv = iL2_max * np.cos(w * tch_lv)

    vlv_ch = iL2_max / w / CLV * (np.sin(w * t2_lv) - np.sin(w * tch_lv))
    vlv_t2 = vlv_n1 + vlv_ch

    vlv_disch = vlv_t2 * (1 - np.exp(-tdisch_lv / RLV / CLV))
    dvlv = vlv_ch - vlv_disch  # Calculate new dvlv
    VLV_min_new += dvlv  # Update VLV_min_new

Power_lv_stable = VLV_min_new**2 / RLV
VLV_stable=VLV_min_new

# Calculate VLV_min_new, ensuring it iterates according to rules with N0_lower_manual
VLV_min_new = VLV_min  # Initialize VLV_min_new
VAC_conducted = VLV_min+vdrop_lv  # Initialize VLV_min_new


num_iterations = int(N0_lower_manual * 1)  # Calculate number of iterations, increasing by 0.5 each time
vlv_min_list = [VLV_min_new]  # List to store VLV_min_new for each iteration
VAC_list = [VAC_conducted]  # List to store VAC_conducted for each iteration









#################################################################################################################






for _ in range(num_iterations):
    vlv_n1 = VLV_min_new  # Update vlv_n1 to current VLV_min_new
    
    #iL2_test = il2_positive_peaks[i % len(il2_positive_peaks)]
    #iL2_test = il2_positive_peaks[2]
    iL2_test = iL2_max
    
    term1 = (vlv_n1 - vcross + vdrop_lv * 1) * w * C2 / (iL2_test * cpar_coeff)
    tch_lv = np.arcsin(term1) / w
    Il2_tchlv = iL2_test * np.cos(w * tch_lv)

    vlv_ch = iL2_test / w / CLV * (np.sin(w * t2_lv) - np.sin(w * tch_lv))
    vlv_t2 = vlv_n1 + vlv_ch

    vlv_disch = vlv_t2 * (1 - np.exp(-tdisch_lv / RLV / CLV))
    dvlv = vlv_ch - vlv_disch  # Calculate new dvlv
    VLV_min_new += dvlv  # Update VLV_min_new
    VAC_conducted = vdrop_lv+VLV_min_new
    
    vlv_min_list.append(VLV_min_new)  # Record VLV_min_new
    VAC_list.append(VAC_conducted)  # Record VLV_min_new

# Calculate the average of VLV_min_new
VLV_min_avg = np.mean(vlv_min_list)


print("VLV_min_new for all iterations:", vlv_min_list)
print("VAC_new for all iterations:", VAC_list)
print("Average of VLV_min_new:", VLV_min_avg)

VLV_N0_manual = VLV_min_new
Power_lv_N0 = VLV_min_avg**2 / RLV
PCE_NRZ = Power_lv_N0 / Power_lv_stable

vlv_n1=VLV_min
term1=(vlv_n1-vcross+vdrop_lv*1)*w*C2/(iL2_max*cpar_coeff)
tch_lv=np.arcsin(term1)/w
Il2_tchlv=iL2_max*np.cos(w*tch_lv)


T=1/f
t2_lv=T/4
t1_lv=tch_lv
dT_lv=t2_lv-t1_lv
Il2_t1_lv=iL2_max*np.cos(w*t1_lv)
Il2_t2_lv=iL2_max*np.cos(w*t2_lv)



vlv_ch=iL2_max/w/CLV*( np.sin(w*t2_lv)- np.sin(w*t1_lv)) 
vlv_t1=VLV_min
vlv_t2=vlv_t1+vlv_ch

tdisch_lv=t1_lv+T/4
vlv_disch=vlv_t2*(1-e**(-tdisch_lv/RLV/CLV))
dvlv=vlv_ch-vlv_disch
vlv_t3=vlv_t1+dvlv
t3_lv=T/2+t1_lv


VLV_min_new=VLV_min+dvlv

########################################################################################################################


RSW=2.21


Vs = 0.29
V21 = Vs * w * M12 / (R1 + (w**2 * M12**2) / (R2 + RSW))

Req=R2+RSW
Q2eq=w*L2/Req
#Q2eq=Q2
V21_q=Vs*k12*sqrt(Q1*Q2eq)*sqrt(Req)/(1+k12**2*Q1*Q2eq)/sqrt(R1)   
V21=V21_q
# Define a
#a = R2 / (2 * L2)  # Calculate a
a = (R2+RSW) / (2 * L2)  # Calculate a

# Define time range, modified from 0 to 1 microsecond
t = np.linspace(0, len(vlv_min_list) / f, 1000)  # From 0 to 1 microsecond

# Define IL2
#IL2 = (abs(V21) * #       ( (-e**(-t * (a + w)) * (-1 + e**(2 * t * w)))*1 + 2 * np.sin(w * t)) ) / (2 * (Req))

Req=R2+RSW
#est=cmath.sqrt(a**2-w**2)


#test=sqrt(w**2-a**2)
#A=-(e**(-t)*(a+test)*(-1+e**(2*t*test))*w)/test
#bracket=A + 2 * np.sin(w * t)
#IL2 = abs(V21) * bracket / (2 * (Req))


#wd=cmath.sqrt(a**2-w**2)
wd=sqrt(w**2-a**2)

bracket=-w*abs(V21)/Req/wd
IL2 = abs(V21)/Req *np.sin(w*t)+ e**(-a*t)*np.sin(wd*t)*bracket*1
#IL2 = abs(V21)/Req *np.sin(w*t)  #stable state
print('IL2_MAX=',max(IL2))
  
#########################################################################################################################

print('VL_k_1=',VL_k_1+vcross)
#print('t1=',t1)
print('t1_nodamping=',t1_nodamping)
print('t2_nodamping=',t2_nodamping)

#print('delta_VHV_nodamping=',delta_VHV_nodamping)
#print('iL2_t_damping=',iL2_t_damping)
print('iL2_t_nodamping=',iL2_t_nodamping)
#print('iL2_t_linear=',iL2_t_linear)
#print('iL2_t_damping_t2_end=',iL2_t_damping_t2_end)
print('iL2_t2_end=',iL2_t2_end)

print('')
print('')
#print('delta_VHV_linear_tcasi=',delta_VHV_linear_tcasi)
print('delta_VHV_linear_my=',delta_VHV_linear_my)
print('VHV_dec=',VHV_dec)
print('')
print('')


#print('VLV_dec=',VLV_dec)
#print('Vripple=',Vripple)
#print('Vripple_approximation=',Vripple_approximation)
print('Vripple_fall=',Vripple_fall)
print('Vripple_rise=',Vripple_rise)
print('')
print('')

#print('VLV_min=',VLV_min)
print('Vripple_VLV=',Vripple_VLV)
#print('VLV_tlv=',VLV_tlv)

print('N0_lower=',N0_lower)

print('')
print('')

print('#####################')

print('vlv_ch=',vlv_ch)
print('vlv_disch=',vlv_disch)

print('################')
print('')

print('dvlv=',dvlv)
print('t1_lv=',t1_lv)
print('t2_lv=',t2_lv)
print('t3_lv=',t3_lv)
print('dT_lv=',dT_lv)
print('Il2_t1_lv=',Il2_t1_lv)
print('Il2_t2_lv=',Il2_t2_lv)
print('VLV_min=',VLV_min)
print('VLV_min_new=',VLV_min_new)
print('VLV_stable=',VLV_stable)
#print('N0_lower_manual=',N0_lower_manual)
#print('VLV_N0_manual=',VLV_N0_manual)
#print('PCE_NRZ=', PCE_NRZ)

print('')
print('')
print('')
print('')


# Generate VLV time vector (corresponding to number of iterations), in µs
t_vlv = np.linspace(0, len(vlv_min_list) / f * 1e6, len(vlv_min_list))  # µs
t_il2 = t * 1e6  # µs



ploss_r_list = []
ploss_c_list = []
pin_list = []
pce_list = []
pout_list = []

#Ploss_c_current = Cpar * np.mean(vlv_min_list)**2 * 2 * f


Cgp=631e-15
Vhv=2.5
#Vhv=2.753


Ploss_c_current = Cpar * Vhv*VAC_conducted * 1 * f
Ploss_c_current_gate=Cgp * (vlv_min_list[0])**2 * 2 * f

#Ploss_c_current = 4000e-6
ploss_c_list.append(Ploss_c_current)

# Iterate through each value in vlv_min_list for calculations
for vlv_val in vlv_min_list:
    
    term1=(vlv_val-vcross+vdrop_lv*1)*w*C2/(iL2_max*cpar_coeff)
    tch_lv=np.arcsin(term1)/w
    Dcharge =tch_lv
    T_half_period = 1 / f / 2 # Renamed T to T_half_period to avoid conflict with T (period)
    Ta = (T_half_period - Dcharge * T_half_period) / 2
    Td = (T_half_period + Dcharge * T_half_period) / 2
    Ron = RLV * (1 / pi * (np.cos(Ta * pi / T_half_period) - np.cos(Td * pi / T_half_period)) / np.sin(Ta * pi / T_half_period) - Dcharge)

    
    
    
    Ploss_r_current = (vlv_val / RLV / Dcharge)**2 * Dcharge * Ron
    #
    #Ploss_c_current = 100e-6
    
    # Use Power_lv_N0 (based on VLV_min_avg) for consistency with average power request
    Power_lv_N0 = vlv_val**2 / RLV
    Pin_current = Ploss_r_current*1 + Ploss_c_current_gate * 1 + Power_lv_N0 
    PCE_current = Power_lv_N0 / Pin_current * 100

    ploss_r_list.append(Ploss_r_current)
    #
    pin_list.append(Pin_current)
    #pce_list.append(PCE_current)
    pout_list.append(Power_lv_N0)

# Calculate average values

#pin_list.append(Ploss_c_current)

pin_list[0]=pin_list[0]+Ploss_c_current*1

average_ploss_r = np.mean(ploss_r_list)
average_ploss_c = np.mean(ploss_c_list)
average_pin = np.mean(pin_list)

average_pout = np.mean(pout_list)

Pin=average_pin
PCE=average_pout/Pin*100/100

#print('PCE=', PCE)
#print('Pout=', average_pout)
#print('Pin=', average_pin)















# Create main figure and right y-axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Left Y-axis: Plot IL2(t)
color1 = 'tab:blue'
ax1.set_xlabel('Time (µs)')
ax1.set_ylabel('$I_{L2}(t)$ (A)', color=color1)
ax1.plot(t_il2, IL2, label='$I_{L2}(t)$', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True)

# Right Y-axis: Plot VLV(t)
ax2 = ax1.twinx()
color2 = 'tab:green'
ax2.set_ylabel('$V_{LV}(t)$ (V)', color=color2)
ax2.plot(t_vlv, vlv_min_list, label='$V_{LV}(t)$', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

# Set plot title
plt.title('$I_{L2}(t)$ and $V_{LV}(t)$ envelope vs Time')

# Add text annotations for PCE, Pout, and Pin
text_str = (
    f'PCE = {PCE:.2%}\n'  # Formats as percentage
    f'Pout = {average_pout*1000:.2f} mW\n' # Converts to mW
    f'Pin = {average_pin*1000:.2f} mW'     # Converts to mW
)
# Adjust the coordinates (0.05, 0.95) to position the text box
# 0.05 is 5% from the left edge, 0.95 is 95% from the bottom edge (top of the plot)
# transform=ax1.transAxes makes the coordinates relative to the axes
ax1.text(0.05, 0.95, text_str, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))


fig.tight_layout()
plt.show()
