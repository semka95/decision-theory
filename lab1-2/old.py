# old approach

def satisfy(consumer_idx, cp_cons, cp_supp):
    supplier_index = 0
    while supplier_index < len(cp_supp):
        consumer = cp_cons[consumer_idx]
        supplier = cp_supp[supplier_index]
        if consumer==0: 
            break
        if supplier==0:
            supplier_index += 1
            continue
        capacity[supplier_index][consumer_idx] = consumer if supplier > consumer else supplier
        cp_cons[consumer_idx] -= capacity[supplier_index][consumer_idx]
        cp_supp[supplier_index] -= capacity[supplier_index][consumer_idx]
        supplier_index += 1
        

consumer_index = 0
while (consumer_index < len(consumers)) and (np.count_nonzero(capacity)!= len(consumers) + len(suppliers) - 1):
    cp_cons = np.copy(consumers)
    cp_supp = np.copy(suppliers)
    satisfy(consumer_index, cp_cons, cp_supp)
    idx = 0
    while idx < len(cp_cons):
        if idx != consumer_index: satisfy(idx, cp_cons, cp_supp)
        idx += 1
    if np.count_nonzero(capacity) == len(cp_cons) + len(cp_supp) - 1: break
    print("building new plan. old plan:")
    print(capacity)
    capacity = np.zeros(costs.shape)
    consumer_index += 1
