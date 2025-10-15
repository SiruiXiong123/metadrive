import math

def calculate_ttc_collision_risk(other_vehicles, penalty_weight=10.0, tau=2.0, max_risk=1.5):
    individual_risks = []
    ttc_values = []
    
    for vehicle_info in other_vehicles:
        if not vehicle_info.get('is_valid', False):
            individual_risks.append(0.0)
            continue
            
        distance = vehicle_info.get('relative_pos_x', 0.0)
        relative_speed = vehicle_info.get('relative_vel_x', 0.0)
        
        if distance <= 0 or relative_speed >= 0:
            individual_risks.append(0.0)
            continue
            
        relative_speed_ms = abs(relative_speed) * 1000 / 3600
        ttc = distance / (relative_speed_ms + 0.01)
        ttc_values.append(ttc)
        
        risk = math.exp(-ttc / tau)
        individual_risks.append(risk)
    
    total_risk_raw = sum(individual_risks)
    total_risk = min(max_risk, total_risk_raw)
    penalty = total_risk 
    
    return penalty