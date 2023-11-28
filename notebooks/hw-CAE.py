# %% [markdown]

"""
(電腦輔助機構分析): 四連桿組，桿2為輸入桿，已知
$$
ω_2 = 100 rpm, r1=6.0, r2=2.0, r3=5.0, r4=5.0, 

$$
使用電腦輔助位置分析試寫一電腦程式分析當0<=θ2<=360。時，θ3和θ4的角度變化
"""

# %%
# 0. 設定參數

omega_2: float = 100.0  # rpm
r1: float = 6.0 # m
r2: float = 2.0 # m
r3: float = 5.0 # m
r4: float = 5.0 # m

# 精度
precision: float = 1e-2

import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt

# 1. 設定輸入角速度
omega_2 = 100 # rpm
omega_2 = omega_2 * 2 * pi / 60 # rad/s

# 2. 設定輸入角度
theta_2 = np.linspace(0, 2 * pi, 1000)

# %%

# 3. 誤差值

def epsilon_1(theta_2: float, theta_3: float, theta_4: float) -> float:
    return (
          r2 * cos(theta_2)
        + r3 * cos(theta_3)
        - r4 * cos(theta_4)
        - r1
    )

def epsilon_2(theta_2: float, theta_3: float, theta_4: float) -> float:
    return (
          r2 * sin(theta_2)
        + r3 * sin(theta_3)
        - r4 * sin(theta_4)
    )

# 4. 誤差修正值

def delta_theta_3(theta_3: float, theta_4: float) -> float:
    e1 = epsilon_1(theta_3, theta_3, theta_4)
    e2 = epsilon_2(theta_3, theta_4)
    return (
        (e1 * cos(theta_3) + e2 * sin(theta_3))
        / sin(theta_3 - theta_4)
    )

def delta_theta_4(theta_3: float, theta_4: float) -> float:
    e1 = epsilon_1(theta_3, theta_4)
    e2 = epsilon_2(theta_3, theta_4)
    return (
        (e1 * cos(theta_4) + e2 * sin(theta_4))
        / sin(theta_3 - theta_4)
    )
# %%

# 5. 
e1 
while 
