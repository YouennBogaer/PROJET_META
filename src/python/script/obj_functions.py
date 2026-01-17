import numpy as np

def obj_functions(chromosome, p, task):
    """
    Calculates the three objective functions: Time, Energy, and Load Balance.
    """
    N = len(chromosome) // 2
    M = p['M']

    # 1. Completion Time (F1)
    TT_task = []

    for i in range(N):
        srv_id = int(chromosome[2 * i])
        allocated_cpu = int(chromosome[2 * i + 1]) # Number of VMs

        size_request = task['G'][i]
        size_result = task['RG'][i]
        parent_mec = task['parent'][i]

        # Determine if task is local, neighbor, or cloud
        if srv_id == 0:
            j = 2 # Cloud
        elif srv_id == parent_mec:
            j = 0 # Local
        else:
            j = 1 # Neighbor

        # Transmission time based on location
        if j == 0:
            t_trans = size_request / p['WKp']
            t_ret = size_result / p['WKp']
        elif j == 1:
            t_trans = (size_request / p['WKp']) + (size_request / p['WKq'])
            t_ret = (size_result / p['WKp']) + (size_result / p['WKq'])
        else:
            t_trans = (size_request / p['WKp']) + (size_request / p['WKq']) + (size_request / p['WKr'])
            t_ret = (size_result / p['WKp']) + (size_result / p['WKq']) + (size_result / p['WKr'])

        # Processing time
        t_calc = size_request / (allocated_cpu * p['Pu'])

        # Total time for this specific task
        t_total = t_trans + t_ret + t_calc
        TT_task.append(t_total)

    # Final F1 is the maximum time among all tasks
    f1_ttot = max(TT_task)

    # 2. Energy Consumption (F2)
    # Basic operating energy for all MEC servers
    ecb = M * p['rx'] * f1_ttot

    ecf = 0
    # Dynamic energy for active VMs on MEC servers only
    for n in range(N):
        srv_id = int(chromosome[2 * n])
        allocated_cpu = int(chromosome[2 * n + 1])
        if srv_id > 0:
            t_calc_n = task['G'][n] / (allocated_cpu * p['Pu'])
            ecf += p['rf'] * allocated_cpu * t_calc_n

    # Energy consumed by idle/unused CPUs
    active_work_time = ecf / p['rf'] if p['rf'] != 0 else 0
    total_available_time = sum(p['mnv'] for _ in range(M)) * f1_ttot
    eck = p['rn'] * (total_available_time - active_work_time)

    # Total Energy
    f2_ectot = ecb + ecf + eck

    # 3. Load Balancing (F3)
    # Calculate VM usage per MEC server
    v_m = []
    for s_idx in range(1, M + 1):
        vms_used = sum(chromosome[2 * k + 1] for k in range(N) if chromosome[2 * k] == s_idx)
        v_m.append(vms_used)

    # Variance of the load across servers
    f3_msu = np.var(v_m)

    return [f1_ttot, f2_ectot, f3_msu]