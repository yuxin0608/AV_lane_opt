import os
# (1) Suppress Qt font errors by switching backend to TkAgg
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Tuple

# === Arrival rate u(t) definition (piecewise) ===
def piecewise_u(t, a_va=1.0):
    if   0.0 <= t <= 0.5:   return 2500 * a_va
    elif 0.5 < t <= 1.0:    return (3000 * t + 1000) * a_va
    elif 1.0 < t <= 2.0:    return 4000 * a_va
    elif 2.0 < t <= 2.5:    return (-3000 * t + 10000) * a_va
    elif 2.5 < t <= 3.0:    return 2500 * a_va
    else:                   return 0.0

@dataclass
class AVLaneReservationSimulator:
    p: float                         # AV penetration rate
    n: int                           # number of time steps
    merg: float                      # merge length (mile)
    a_va: float                      # aggressiveness factor for u(t)
    vc: float                        # cruise speed (mph)
    conversion_factor: float         # 5280 / 3600**2
    fix: bool                        # fixed-capacity flag
    u_func: Callable[[float], float] # arrival rate function λ(t)
    
    # constants
    k_hd: int = 1023                 
    k_av: int = 2018
    a_av: float = 2.0
    a_hdv: float = 1.5
    back_table: dict = field(default_factory=lambda: {
        0.1:0.02, 0.2:0.06, 0.3:0.14, 0.4:0.22, 0.5:0.30,
        0.6:0.38, 0.7:0.46, 0.8:0.54, 0.9:0.77, 1.0:1.00
    })
    
    def __post_init__(self):
        key = round(self.p, 1)
        f = self.back_table.get(key)
        if f is None:
            raise ValueError("p must be in 0.1 increments (0.1–1.0)")
        self.back_new  = 8 + f * (12 - 8)
        self.delta_new = (1 + self.back_new/20) * np.log(1 + 20/self.back_new) - 1
        self.cap_new   = self.k_hd + f*(self.k_av - self.k_hd)

    # Effective capacity modifiers
    def term1(self, t: float) -> float:
        return self.k_hd * (1 - self.conversion_factor*self.merg*self.u_func(t)*self.vc *
                            ((self.p/(4*self.a_av)) + (1-self.p)/(2*self.a_hdv)))
    def term2(self, t: float) -> float:
        return self.k_av * (1 - self.conversion_factor*self.merg*self.u_func(t)*self.vc *
                            (self.p/(6*self.a_av)))

    # Discharge rate for HDV lane with queue logic
    def a(self, t: float) -> float:
        base = self.u_func(t)*(1-self.p)
        if self.fix:
            # fixed branch unchanged
            return self.k_hd if base >= self.k_hd else base
        # not fix: check queue
        if self._q[-1] > 0:
            # congested: threshold = term1
            threshold = self.term1(t)
        else:
            # no queue: threshold = k_hd
            threshold = self.k_hd
        return threshold if base >= threshold else base

    # Discharge rate for AV lane with queue logic
    def b(self, t: float) -> float:
        base = self.u_func(t)*self.p
        if self.fix:
            return self.k_av if base >= self.k_av else base
        if self._q[-1] > 0:
            threshold = self.term2(t)
        else:
            threshold = self.k_av
        return threshold if base >= threshold else base

    # Combined single-lane discharge when AV lane is off, with queue logic
    def discount(self, t: float) -> float:
        return 1 - self.u_func(t)*self.merg*self.delta_new*self.conversion_factor* \
               self.back_new*(1/(2*self.a_hdv) + self.p/(2*self.a_av) - self.p/(2*self.a_hdv))
    def c(self, t: float) -> float:
        base = self.u_func(t)
        if self.fix:
            thresh = 2*self.cap_new
        else:
            # not fix: choose threshold based on queue
            if self._q[-1] > 0:
                thresh = 2*self.discount(t)*self.cap_new
            else:
                thresh = 2*self.cap_new
        return thresh if base >= thresh else base

    # Compute queue-based delay and update internal queue
    def delay(self, ta: float, tb: float) -> float:
        time = np.linspace(0, 3, self.n)
        dt = time[1] - time[0]
        # initialize internal queue list
        self._q = [0.0]
        for t in time[1:]:
            if t < ta or t > tb:
                inflow = self.u_func(t) - self.c(t)
            else:
                inflow = (self.u_func(t)*(1-self.p) - self.a(t) +
                          self.u_func(t)*self.p       - self.b(t))
            # update queue
            self._q.append(max(self._q[-1] + inflow*dt, 0.0))
        return sum(self._q)*dt

    # Optimize for best (ta, tb)
    def optimize(self, ta_range: np.ndarray, tb_range: np.ndarray) -> Tuple[float, float, float]:
        best_d, best_ta, best_tb = float('inf'), 0.0, 0.0
        for ta in ta_range:
            for tb in tb_range:
                if ta <= tb:
                    d = self.delay(ta, tb)
                    if d < best_d:
                        best_d, best_ta, best_tb = d, ta, tb
        return best_ta, best_tb, best_d

    # Generate discharge series for plotting
    def discharge_series(self, ta: float, tb: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        time = np.linspace(0, 3, self.n)
        hdv = np.zeros_like(time)
        av  = np.zeros_like(time)
        for i, t in enumerate(time):
            if t < ta or t > tb:
                val = self.c(t) / 2.0
                hdv[i] = val
                av[i]  = val
            else:
                hdv[i] = self.a(t)
                av[i]  = self.b(t)
        return time, hdv, av
        
    # def capacity_series(self, ta: float, tb: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     """
    #     Generate HDV and AV lane capacity series with queue logic and activation window [ta,tb]:
    #       - outside [ta,tb]: total capacity = c(t), split evenly
    #       - inside  [ta,tb]: for each lane, if q>0 then threshold=term,
    #                         else threshold=k; then cap = max(threshold, base)
    #     Returns time, cap_hd, cap_av.
    #     """
    #     time = np.linspace(0, 3, self.n)
    #     dt   = time[1] - time[0]
    #     cap_hd = np.zeros_like(time)
    #     cap_av = np.zeros_like(time)
    #     q = 0.0

    #     for i, t in enumerate(time):
    #         u_t   = self.u_func(t)
    #         base_h = u_t * (1 - self.p)
    #         base_a = u_t * self.p

    #         if t < ta or t > tb:
    #             # AV lane deactivated: shared capacity c(t)
    #             total = self.c(t)
    #             cap_hd[i] = total / 2.0
    #             cap_av[i] = total / 2.0
    #         else:
    #             # AV lane activated: per‐lane capacity with queue logic
    #             # HDV lane
    #             if q > 0:
    #                 thresh_h = self.term1(t)
    #             else:
    #                 thresh_h = self.k_hd
    #             cap_hd[i] = thresh_h if base_h >= thresh_h else base_h

    #             # AV lane
    #             if q > 0:
    #                 thresh_a = self.term2(t)
    #             else:
    #                 thresh_a = self.k_av
    #             cap_av[i] = thresh_a if base_a >= thresh_a else base_a

    #         # update queue backlog
    #         net = u_t - (cap_hd[i] + cap_av[i])
    #         q = max(q + net * dt, 0.0)

    #     return time, cap_hd, cap_av
    def capacity_series(self, ta: float, tb: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate HDV and AV lane capacity series with queue logic and activation window [ta,tb]:
        - outside [ta,tb]: if q>0 → total = 2 * discount(t) * cap_new; else → total = 2 * cap_new.
                            Then split evenly between the two lanes.
        - inside  [ta,tb]: for each lane,
                if q>0: threshold = term; else threshold = k;
                if base < threshold → capacity = k; else → capacity = term.
        Returns time, cap_hd, cap_av.
        """
        time = np.linspace(0.01, 3, self.n)
        dt   = time[1] - time[0]
        cap_hd = np.zeros_like(time)
        cap_av = np.zeros_like(time)
        q = 0.0

        for i, t in enumerate(time):
            u_t    = self.u_func(t)
            base_h = u_t * (1 - self.p)
            base_a = u_t * self.p

            if t < ta or t > tb:
                # AV lane deactivated: shared capacity
                if q > 0:
                    total = 2 * self.discount(t) * self.cap_new
                else:
                    total = 2 * self.cap_new
                cap_hd[i] = total / 2.0
                cap_av[i] = total / 2.0
            else:
                # AV lane activated: per‐lane capacity with queue logic
                if q > 0:
                    thresh_h = self.term1(t)
                    thresh_a = self.term2(t)
                else:
                    thresh_h = self.k_hd
                    thresh_a = self.k_av

                # HDV lane: if base_h < thresh_h → capacity = k_hd; else = term1
                if base_h < thresh_h:
                    cap_hd[i] = self.k_hd
                else:
                    cap_hd[i] = self.term1(t)

                # AV lane: if base_a < thresh_a → capacity = k_av; else = term2
                if base_a < thresh_a:
                    cap_av[i] = self.k_av
                else:
                    cap_av[i] = self.term2(t)

            # Update queue backlog
            net = u_t - (cap_hd[i] + cap_av[i])
            q   = max(q + net * dt, 0.0)

        return time, cap_hd, cap_av

    # Plot discharge with distinct linestyles
    def plot_discharge(self, ta: float, tb: float, ax: plt.Axes):
        time, hdv, av = self.discharge_series(ta, tb)
        ax.plot(time, hdv, label="HDV lane", linestyle='-')
        ax.plot(time, av,  label="AV lane", linestyle='--')
        ax.axvline(ta, color='red', linestyle='--')
        ax.axvline(tb, color='red', linestyle='--')
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Discharge rate (veh/h)")
        ax.set_xlim(0, 3)
        ax.legend()
        ax.grid(True)

# def main():
#     # Common parameters
#     params = dict(
#         p=0.1,
#         n=100,
#         merg=0.20,
#         a_va=1.0,
#         vc=20,
#         conversion_factor=5280/(3.28*3600**2),
#         u_func=lambda t: piecewise_u(t, a_va=1.0)
#     )
#     # Instantiate simulators
#     sim_fix   = AVLaneReservationSimulator(fix=True,  **params)
#     sim_eff   = AVLaneReservationSimulator(fix=False, **params)
    
#     # Search ranges
#     ta_range = np.arange(0, 3.01, 0.01)
#     tb_range = np.arange(0, 3.01, 0.01)
    
#     # Optimize both scenarios
#     ta_fix, tb_fix, d_fix = sim_fix.optimize(ta_range, tb_range)
#     ta_eff, tb_eff, d_eff = sim_eff.optimize(ta_range, tb_range)
#     fix_eff_delay = sim_eff.delay(ta_fix, tb_fix)
    
#     print(f"Fixed scenario:     ta={ta_fix:.2f}, tb={tb_fix:.2f}, delay={d_fix:.2f}")
#     print(f"Effective scenario: ta={ta_eff:.2f}, tb={tb_eff:.2f}, delay={d_eff:.2f}")
#     print(f"Fixed scenario effective delay:     delay={fix_eff_delay:.2f}\n")
    
#     # (2) Print every 15-minute (0.25h) discharge rates
#     t_points = np.arange(0, 3.1, 0.1)
#     print("== Fixed capacity discharge @ 15-min intervals ==")
#     for t in t_points:
#         if t < ta_fix or t > tb_fix:
#             rate = sim_fix.c(t)/2
#             hdv = av = rate
#         else:
#             hdv = sim_fix.a(t)
#             av  = sim_fix.b(t)
#         print(f"t={t:.2f}h | HDV: {hdv:.2f} veh/h, AV: {av:.2f} veh/h")
    
#     print("\n== Effective capacity discharge @ 15-min intervals ==")
#     for t in t_points:
#         if t < ta_eff or t > tb_eff:
#             rate = sim_eff.c(t)/2
#             hdv = av = rate
#         else:
#             #ifnot = sim_eff.c(t)/2
#             hdv = sim_eff.a(t)
#             av  = sim_eff.b(t)
#         print(f"t={t:.2f}h | HDV: {hdv:.2f} veh/h, AV: {av:.2f} veh/h")

#     ######### discharge rate
#     # Plot total road discharge for effective scenario only 
#     # time, hdv, av = sim_eff.discharge_series(ta_eff, tb_eff)
#     # road = hdv + av
#     # fig, ax = plt.subplots(figsize=(10, 4))
#     # ax.plot(time, road, label="Road discharge rate", linestyle='-')
#     # #ax.axvline(ta_eff, color='red', linestyle='--')
#     # #ax.axvline(tb_eff, color='red', linestyle='--')
#     # ax.set_xlabel("Time (h)")
#     # ax.set_ylabel("Discharge rate (veh/h)")
#     # ax.set_title(f"Effective Road Discharge (ta={ta_eff:.2f}, tb={tb_eff:.2f})")
#     # ax.legend()
#     # ax.grid(True)
#     # plt.tight_layout()
#     # plt.show()

#     ######### capacity
#     time, cap_hd, cap_av = sim_eff.capacity_series(ta_eff, tb_eff)
#     road_cap = cap_hd + cap_av
#     plt.figure(figsize=(10,4))
#     plt.plot(time, road_cap, label="Road Capacity", linewidth=2)
#     #plt.axhline(sim_eff.k_hd+sim_eff.k_av, color='grey', linestyle='--', label="Max base capacity")
#     plt.xlabel("Time (h)")
#     plt.ylabel("Capacity (veh/h)")
#     plt.title(f"Effective Road Capacity over Time (p = {params['p']})")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()


# def main():
#     print("Start simulation...")

#     # 公共参数（除了 p 以外）
#     base_params = dict(
#         n=100,
#         merg=0.20,
#         a_va=1.0,
#         vc=20,
#         conversion_factor=5280/(3.28*3600**2),
#         u_func=lambda t: piecewise_u(t, a_va=1.0)
#     )

#     # 搜索范围（可以先用 0.05 调试，确认没问题再改回 0.01）
#     ta_range = np.arange(0, 3.01, 0.01)
#     tb_range = np.arange(0, 3.01, 0.01)

#     # 想比较的三种穿透率
#     p_list = [0.7, 0.4, 0.1]

#     plt.figure(figsize=(10, 4))

#     for p in p_list:
#         print(f"\n=== Start p = {p} ===")
#         # 为当前 p 组装参数
#         params = dict(base_params)   # 拷贝一份
#         params["p"] = p

#         # 只需要有效容量场景（fix=False）
#         sim_eff = AVLaneReservationSimulator(fix=False, **params)

#         # 优化当前 p 下的 ta, tb
#         ta_eff, tb_eff, d_eff = sim_eff.optimize(ta_range, tb_range)
#         print(f"p={p:.1f} | ta={ta_eff:.2f}, tb={tb_eff:.2f}, delay={d_eff:.2f}")

#         # 计算该 p 下的 capacity 曲线
#         time, cap_hd, cap_av = sim_eff.capacity_series(ta_eff, tb_eff)
#         road_cap = cap_hd + cap_av

#         # 画在同一张图上
#         plt.plot(time, road_cap, label=f"p = {p:.1f}")

#     # 统一设置图像属性
#     plt.xlabel("Time (h)")
#     plt.ylabel("Capacity (veh/h)")
#     plt.title("Effective Road Capacity over Time for Different AV Penetration Rates")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

def main():
    print("Start simulation...")

    # search range
    ta_range = np.arange(0, 3.01, 0.01)
    tb_range = np.arange(0, 3.01, 0.01)

    # loop over control factors (a_va)
    control_factors = np.arange(0.7, 1.51, 0.1)

    print("\nControlFactor | LowestCapacity | ActivateSlot | OptimalDelay") 
    print("-------------------------------------------------------------")

    for cf in control_factors:
        # build parameters for this cf
        params = dict(
            p=0.7,
            n=100,
            merg=0.20,
            a_va=cf,                     # <--- varying a_va
            vc=20,
            conversion_factor=5280/(3.28*3600**2),
            u_func=lambda t, c=cf: piecewise_u(t, a_va=c)   # <--- must bind cf
        )

        # simulators
        sim_eff = AVLaneReservationSimulator(fix=False, **params)

        # optimize window
        ta_eff, tb_eff, d_eff = sim_eff.optimize(ta_range, tb_range)

        # compute capacity time series
        time, cap_hd, cap_av = sim_eff.capacity_series(ta_eff, tb_eff)
        road_cap = cap_hd + cap_av

        # lowest effective capacity
        lowest_cap = np.min(road_cap)

        # activation slot string
        if ta_eff == 0 and tb_eff == 0:
            slot_str = "None"
        else:
            slot_str = f"[{ta_eff:.2f}, {tb_eff:.2f}]"

        # print result
        print(f"{cf:.1f} | {lowest_cap:.0f} | {slot_str} | {d_eff:.0f}")

    print("\nDone.")


if __name__ == "__main__":
    main()