import numpy as np
import pandas as pd

def bootstrap_acn_to_tensor(acn_sessions_df: pd.DataFrame, num_samples: int = 5000, n_nodes: int = 32) -> np.ndarray:
    """
    Transforms raw ACN charging sessions into a (num_samples, 24, n_nodes) tensor
    using bootstrapping and traffic-weighted spatial distribution.
    
    Expected columns in acn_sessions_df: ['arrival_hour', 'energy_delivered_kwh']
    """
    print(f"Bootstrapping {num_samples} daily scenarios from {len(acn_sessions_df)} historical sessions...")
    
    # 1. Initialize the empty tensor (Samples x 24 hours x 32 nodes)
    demand_tensor = np.zeros((num_samples, 24, n_nodes))
    
    # 2. The Traffic Weighting (Spatial Distribution)
    # Sujay will replace this mock array with his ACTUAL Census/OSMnx Traffic Index
    # Higher index = busier intersection = higher probability of an EV parking there
    mock_traffic_index = np.random.beta(a=2.0, b=5.0, size=n_nodes) 
    traffic_probs = mock_traffic_index / np.sum(mock_traffic_index) # Normalize to 1.0
    
    # 3. Bootstrapping Loop
    # Assume Boulder sees an average of ~400 charging sessions a day across this grid
    sessions_per_day_mean = 400 
    
    for i in range(num_samples):
        # Add real-world variance (e.g., weekends vs weekdays)
        daily_session_count = int(max(50, np.random.normal(sessions_per_day_mean, scale=80)))
        
        # Sample real sessions with replacement (The Bootstrap)
        daily_sessions = acn_sessions_df.sample(n=daily_session_count, replace=True)
        
        for _, session in daily_sessions.iterrows():
            # Extract the real human behavior (when they plug in, how much they need)
            # Ensuring hour is strictly bounded between 0 and 23
            arrival_hour = int(session['arrival_hour']) % 24 
            energy_kwh = float(session['energy_delivered_kwh'])
            
            # Assign the car to a physical node based strictly on the traffic flow
            assigned_node = np.random.choice(np.arange(n_nodes), p=traffic_probs)
            
            # Map the demand spike
            demand_tensor[i, arrival_hour, assigned_node] += energy_kwh
            
    print(f"Extraction Complete! Output Tensor Shape: {demand_tensor.shape}")
    print(f"Zero-Output Check: {np.mean(demand_tensor == 0)*100:.2f}% sparsity (Target: < 95%)")
    
    return demand_tensor

if __name__ == "__main__":
    # --- MOCK USAGE FOR SUJAY TO TEST ---
    # Simulating 10,000 rows of raw Caltech ACN data
    mock_acn_data = pd.DataFrame({
        'arrival_hour': np.random.normal(loc=17, scale=3, size=10000).astype(int), # Peak at 5 PM
        'energy_delivered_kwh': np.random.exponential(scale=20.0, size=10000)      # Avg 20 kWh
    })
    
    # Generate 5,000 days of data mapped to the 32 IEEE buses
    training_tensor = bootstrap_acn_to_tensor(mock_acn_data, num_samples=5000, n_nodes=32)
    
    # Save for the Generative Core
    np.save('bootstrapped_demand_tensor_32nodes.npy', training_tensor)