import scipy.sparse as sparse

def MLPCal(in_size, out_size, n_hidden, hidden_size): 
    count = in_size * hidden_size + hidden_size
    
    for _ in range(n_hidden - 1):
        count += hidden_size * hidden_size + hidden_size

    count += hidden_size * out_size + out_size

    return count

def ParaCal():
    #========================================
    # config from graphnet.py
    e_in = 3
    n_in = 2
    g_in = 0

    e_mid = 4
    n_mid = 4
    g_mid = 0

    e_out = 1
    n_out = 1
    g_out = 0

    n_hidden = 1
    hidden_size = 16

    middle_layer = 2
    #========================================


    edge = MLPCal(e_in+2*n_in+g_in,e_mid,n_hidden,hidden_size)
    node = MLPCal(n_in+e_mid+g_in,n_mid,n_hidden,hidden_size) + MLPCal(n_in+n_mid,n_mid,n_hidden,hidden_size)

    for _ in range(middle_layer):
        edge += MLPCal(e_mid+2*n_mid+g_mid,e_mid,n_hidden,hidden_size)
        node += MLPCal(n_mid+e_mid+g_mid,n_mid,n_hidden,hidden_size) + MLPCal(n_mid+n_mid,n_mid,n_hidden,hidden_size)

    edge += MLPCal(e_mid+2*n_mid+g_mid,e_out,n_hidden,hidden_size)
    node += MLPCal(n_mid+e_out+g_mid,n_out,n_hidden,hidden_size) + MLPCal(n_mid+n_out,n_out,n_hidden,hidden_size)

    total = edge + node 
    return total

num_para = ParaCal()
mem = num_para * 8 /(10**9)
print('gpu memory of parameter is:',mem)
