import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

from data_pipeline.ieee33bus_data import get_network_data

class LinDistFlowLoss(nn.Module):
    """
    Differentiable AC power flow constraints (LinDistFlow) for the IEEE 33-Bus feeder.
    
    Transforms the iterative Forward-Backward Sweep into a set of dense matrix
    multiplications, enabling gradients to flow backward through topological constraints.
    """
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        
        # Load topology from the data pipeline module
        nd = get_network_data()
        
        # We only care about load buses 2..33, so N = 32
        N = 32
        
        # Build the path matrix B [32, 32]
        # B[i, k] = 1 if branch k is on the path from the substation to bus i+2
        B = np.zeros((N, N), dtype=np.float32)
        
        from_bus = nd['from_bus']
        to_bus = nd['to_bus']
        
        # Map child to parent branch index
        par_branch = {}
        parent_bus = {}
        for k in range(N):
            par_branch[to_bus[k]] = k
            parent_bus[to_bus[k]] = from_bus[k]
            
        for i in range(N):
            curr = i + 2
            while curr > 1:
                k = par_branch[curr]
                B[i, k] = 1.0
                curr = parent_bus[curr]
                
        # Register static tensors so PyTorch moves them to the given device
        self.register_buffer("B", torch.from_numpy(B).to(device))
        self.register_buffer("A", torch.from_numpy(B.T).to(device))  # A = B^T
        
        self.register_buffer("R", torch.from_numpy(nd['R']).float().to(device))
        self.register_buffer("X", torch.from_numpy(nd['X']).float().to(device))
        
        # Base loads [32]
        self.register_buffer("base_P", (torch.from_numpy(nd['base_P_mw'][1:]).float() / nd['base_mva']).to(device))
        self.register_buffer("base_Q", (torch.from_numpy(nd['base_Q_mvar'][1:]).float() / nd['base_mva']).to(device))
        
        self.base_mva = float(nd['base_mva'])
        self.xfmr_kva = float(nd['xfmr_kva'])
        self.v_min = float(nd['v_min'])
        self.v_max = float(nd['v_max'])
        self.i_lim_pu = float(nd['i_lim_pu'])

    def forward(self, ev_demand_kw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate LinDistFlow constraints.
        
        Args:
            ev_demand_kw: Tensor of shape [..., 32], active power demand in kW for buses 2..33
                          (The time axis can be combined into the batch axis).
                          
        Returns:
            Tuple of scalar penalty tensors: (voltage_penalty, thermal_penalty, xfmr_penalty)
        """
        # Convert EV demand kW -> MW -> PU
        ev_p_pu = (ev_demand_kw / 1000.0) / self.base_mva
        # Assume 0.9 lagging power factor for the EV charging load: Q = P * tan(acos(0.9)) -> roughly 0.4843 * P
        ev_q_pu = ev_p_pu * 0.484322
        
        # Total load = base + EV
        P_load = self.base_P + ev_p_pu
        Q_load = self.base_Q + ev_q_pu
        
        # 1. Branch Flows (Pbr = P_load @ A^T, ignoring losses per LinDistFlow)
        # Using functional linear: F.linear(input, weight) = input @ weight^T + bias
        # Here, weight = B.T, so input @ B = input @ A^T
        Pbr = torch.nn.functional.linear(P_load, self.A.T)
        Qbr = torch.nn.functional.linear(Q_load, self.A.T)
        
        # 2. Voltage profile (LinDistFlow V^2 drop)
        # Drop = 2 * (R*Pbr + X*Qbr)
        drop = 2.0 * (self.R * Pbr + self.X * Qbr)
        
        # V2 = 1.0 - (drop @ B^T)
        V2 = 1.0 - torch.nn.functional.linear(drop, self.B)
        
        # We penalize bounds on V^2 effectively, or we can use sqrt. Since it's around 1.0, V bounds are similar to V^2 bounds
        V = torch.sqrt(torch.clamp(V2, min=0.25))
        
        v_under = torch.relu(self.v_min - V)
        v_over = torch.relu(V - self.v_max)
        voltage_penalty = torch.mean(v_under**2 + v_over**2)
        
        # 3. Thermal current approximation
        # |I| approx sqrt(Pbr^2 + Qbr^2) since V approx 1.0
        I_pu = torch.sqrt(Pbr**2 + Qbr**2)
        thermal_excess = torch.relu(I_pu / self.i_lim_pu - 1.0)
        thermal_penalty = torch.mean(thermal_excess**2)
        
        # 4. Transformer capacity (Branch 0 flow)
        # S_xfmr_pu = sqrt(Pbr[0]^2 + Qbr[0]^2)
        S_xfmr_pu = torch.sqrt(Pbr[..., 0]**2 + Qbr[..., 0]**2)
        # capacity in pu
        cap_pu = self.xfmr_kva / (self.base_mva * 1000.0)
        xfmr_excess = torch.relu(S_xfmr_pu / cap_pu - 1.0)
        xfmr_penalty = torch.mean(xfmr_excess**2)
        
        return voltage_penalty, thermal_penalty, xfmr_penalty
