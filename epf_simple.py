import math

def calculate_epf_collision_risk(other_vehicles, penalty_weight=10.0, max_risk=1.5):
    """
    基于EPF (Elliptical Potential Field) 的碰撞风险计算函数
    
    参数:
    - other_vehicles: 其他车辆信息列表
    - penalty_weight: 惩罚权重
    - max_risk: 最大风险值上限
    
    返回:
    - penalty: 总的风险惩罚值
    """
    
    # EPF参数设置 (根据论文)
    mu_x = 0.2 * 9.8  # 纵向最大减速度 (m/s^2)
    mu_y = 0.1 * 9.8  # 横向最大减速度 (m/s^2)
    tau = 0.5  # 反应时间参数
    agent_width = 2.0  # 智能体车辆宽度 (m)
    
    total_critical_risk = 0.0
    total_broader_risk = 0.0
    
    for vehicle_info in other_vehicles:
        if not vehicle_info.get('is_valid', False):
            continue
            
        # 获取相对位置和速度
        dx = vehicle_info.get('relative_pos_x', 0.0)  # 纵向相对位置
        dy = vehicle_info.get('relative_pos_y', 0.0)  # 横向相对位置
        
        # 相对速度 (km/h -> m/s)
        dvx_raw = vehicle_info.get('relative_vel_x', 0.0)
        dvy_raw = vehicle_info.get('relative_vel_y', 0.0)
        
        # 转换速度单位
        dvx_ms = dvx_raw * 1000 / 3600  # km/h -> m/s
        dvy_ms = dvy_raw * 1000 / 3600
        
        # === 计算关键特征参数 ===
        
        # 安全距离参数 (公式 3.26)
        safe_distance_y = 0.5 * (agent_width + vehicle_info.get('width', 2.0))
        ac = 1.0  # 最小安全纵向距离
        bc = safe_distance_y + 1.0  # 最小安全横向距离
        
        # 相对速度处理 (公式 3.29, 3.30)
        if dx * dvx_ms < 0:
            dvxb = dvx_ms
        else:
            dvxb = 0.0
            
        if dvy_ms * dy < 0 and abs(dy) > 0.5 * safe_distance_y:
            dvyb = dvy_ms
        else:
            dvyb = 0.0
        
        # a, b 参数计算 (公式 3.27)
        vs = math.sqrt(dvx_ms**2 + dvy_ms**2) + 0.01  # 避免除零
        a = ac + (dvxb**2) / (2 * abs(mu_x)) + tau * vs
        b = bc + (dvyb**2) / (2 * abs(mu_y))
        
        # === Critical Region Field 计算 (公式 3.31) ===
        if a > 0 and b > 0:
            denominator_c = ((dx/a)**4 + (dy/b)**4 + 1)**2
            if denominator_c > 0:
                Ec = 0.5 / denominator_c
            else:
                Ec = 0.0
        else:
            Ec = 0.0
            
        # === Broader Region Field 计算 (公式 3.32) ===
        if a > 0 and b > 0:
            denominator_b = (dx/a)**2 + (dy/b)**2 + 1
            if denominator_b > 0:
                Eb = 0.5 / denominator_b
            else:
                Eb = 0.0
        else:
            Eb = 0.0
        
        # 累积风险
        total_critical_risk += Ec
        total_broader_risk += Eb
    
    # === Total Collision Risk 计算 (公式 3.33) ===
    total_risk_raw = total_critical_risk + total_broader_risk
    total_risk = min(max_risk, total_risk_raw)
    penalty = total_risk 
    return penalty