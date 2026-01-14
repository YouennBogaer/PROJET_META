import numpy as np



def obj_functions(chromosome, p, task) :
    #print(f"Type du premier élément : {type(chromosome[0])}")
    #print(f"Shape du chromosome : {np.shape(chromosome)}")
    N = len(chromosome)//2
    M = p['M']

    TT_task = [] #temps d'éxécution de la task


    #Matrice de zéro pour représenter le temps total d'éxécution pour chaque serveur pour la consomation d'énergie

    T_server = np.zeros(M+1)

    for i in range (N):
        srv_id = int(chromosome[2*i])
        allocated_cpu = int(chromosome[2*i+1]) #Nb VM

        size_request = task['G'][i]
        size_result = task['RG'][i]

        parent_mec = task['parent'][i]

        #détermination de j dans le papier

        if srv_id == 0 :
            j = 2
        elif srv_id == parent_mec :
            j = 1
        else:
            j = 0

        if j == 0:
            t_trans = size_request / p['WKp']
            t_ret = size_result / p['WKp']  # TBn [cite: 346]
        elif j == 1:
            t_trans = (size_request / p['WKp']) + (size_request / p['WKq'])
            t_ret = (size_result / p['WKp']) + (size_result / p['WKq'])  # TBn [cite: 348]
        else:
            t_trans = (size_request / p['WKp']) + (size_request / p['WKq']) + (size_request / p['WKr'])
            t_ret = (size_result / p['WKp']) + (size_result / p['WKq']) + (size_result / p['WKr'])  # TBn [cite: 349]

        t_calc = size_request/(allocated_cpu * p['Pu']) #temps de calcul

        t_total = t_trans + t_ret +t_calc

        TT_task.append(t_total)

        if srv_id > 0 :
            T_server[srv_id] = max(T_server[srv_id], t_total)

    f1_ttot = max(TT_task)


    #Calc de l'NRJ

    #Conso à vide
    ecb = sum(T_server[1:]*p['rx'])

    ecf = 0
    eck = 0

    for s_id in range(1, M+1) :
        vms_used = sum(chromosome[2 * k + 1] for k in range(N) if chromosome[2 * k] == s_id)
        # Énergie des VM occupées sur ce serveur
        ecf += vms_used * T_server[s_id] * p['rf']
        # Énergie des VM libres sur ce serveur
        eck += (p['mnv'] - vms_used) * T_server[s_id] * p['rn']

    f2_ectot = ecb + ecf + eck

    msu_servers = []
    for s_idx in range(1, M + 1):
        vms_used = sum(chromosome[2 * k + 1] for k in range(N) if chromosome[2 * k] == s_idx)
        msu_servers.append(vms_used / p['mnv'])

    # MSU est la variance de ces utilisations
    f3_msu = np.var(msu_servers)

    return [f1_ttot, f2_ectot, f3_msu]
