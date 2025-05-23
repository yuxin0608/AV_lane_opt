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
    
    # constants for capacities
    k_hd: int = 1023
    k_av: int = 2018
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
        self.delta_new = (1 + self.back_new / 20) * np.log(1 + 20 / self.back_new) - 1
        self.cap_new   = self.k_hd + f * (self.k_av - self.k_hd)
    
    # Effective capacity modifiers
    def term1(self, t):
        return self.k_hd * (1 - self.conversion_factor * self.merg * self.u_func(t) * self.vc *
                            ((self.p/12) + (1-self.p)/3))
    
    def term2(self, t):
        return self.k_av * (1 - self.conversion_factor * self.merg * self.u_func(t) * self.vc *
                            (self.p/18))
    
    # Discharge rates for each lane
    def a(self, t):
        base = self.u_func(t) * (1 - self.p)
        return self.k_hd if (self.fix and base >= self.k_hd) else (
               (self.term1(t) if not self.fix and base >= self.k_hd else base))
    
    def b(self, t):
        base = self.u_func(t) * self.p
        return self.k_av if (self.fix and base >= self.k_av) else (
               (self.term2(t) if not self.fix and base >= self.k_av else base))
    
    # Combined single-lane discharge when AV lane is off
    def discount(self, t):
        return 1 - self.u_func(t) * self.merg * self.delta_new * self.conversion_factor * \
               self.back_new * (1/3 + self.p/3 - self.p/6)
    
    def c(self, t):
        base = self.u_func(t)
        if self.fix:
            return 2 * self.cap_new if base >= 2 * self.cap_new else base
        else:
            return 2 * self.discount(t) * self.cap_new if base >= 2 * self.cap_new else base
    
    # Compute queue-based delay
    def delay(self, ta: float, tb: float) -> float:
        time = np.linspace(0, 3, self.n)
        dt = time[1] - time[0]
        q = [0.0]
        for t in time[1:]:
            if t < ta or t > tb:
                net = self.u_func(t) - self.c(t)
            else:
                net = (self.u_func(t)*(1-self.p) - self.a(t) +
                       self.u_func(t)*self.p     - self.b(t))
            q.append(max(q[-1] + net * dt, 0.0))
        return sum(q) * dt
    
    # Optimize for best (ta, tb)
    def optimize(self,
                 ta_range: np.ndarray,
                 tb_range: np.ndarray
                ) -> Tuple[float, float, float]:
        best_d = float('inf')
        best_ta = best_tb = 0.0
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

def main():
    # Common parameters
    params = dict(
        p=1,
        n=1000,
        merg=0.15,
        a_va=1.0,
        vc=20,
        conversion_factor=5280/3600**2,
        u_func=lambda t: piecewise_u(t, a_va=1.0)
    )
    # Instantiate simulators
    sim_fix   = AVLaneReservationSimulator(fix=True,  **params)
    sim_eff   = AVLaneReservationSimulator(fix=False, **params)
    
    # Search ranges
    ta_range = np.arange(0, 3.01, 0.01)
    tb_range = np.arange(0, 3.01, 0.01)
    
    # Optimize
    # ta_fix, tb_fix, d_fix = sim_fix.optimize(ta_range, tb_range)
    ta_eff, tb_eff, d_eff = sim_eff.optimize(ta_range, tb_range)
    # print(f"Fixed:   ta={ta_fix:.2f}, tb={tb_fix:.2f}, delay={d_fix:.2f}")
    print(f"Effective: ta={ta_eff:.2f}, tb={tb_eff:.2f}, delay={d_eff:.2f}")
    
    # Plot results
    # fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    # fig.suptitle(f"Lane Discharge Rates (p = {params['p']})")
    
    # axes[0].set_title(f"Fixed capacity (ta={ta_fix:.2f}, tb={tb_fix:.2f})")
    # sim_fix.plot_discharge(ta_fix, tb_fix, axes[0])
    
    # axes[1].set_title(f"Effective capacity (ta={ta_eff:.2f}, tb={tb_eff:.2f})")
    # sim_eff.plot_discharge(ta_eff, tb_eff, axes[1])
    
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.show()

if __name__ == "__main__":
    main()
