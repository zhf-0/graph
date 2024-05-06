import scipy.sparse as sparse

def MLPInfo(in_size, out_size, n_hidden, hidden_size): 
    count = in_size * hidden_size + hidden_size
    intermid = hidden_size
    
    for _ in range(n_hidden - 1):
        count += hidden_size * hidden_size + hidden_size
        intermid += hidden_size

    count += hidden_size * out_size + out_size
    intermid += out_size

    return count, intermid

def GraphInfo():
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
    edge = 0
    node = 0
    edge_intermid = 0
    node_intermid = 0


    tmp1,tmp2 = MLPInfo(e_in+2*n_in+g_in,e_mid,n_hidden,hidden_size)
    edge += tmp1
    edge_intermid += tmp2

    tmp1,tmp2 = MLPInfo(n_in+e_mid+g_in,n_mid,n_hidden,hidden_size) 
    node += tmp1
    node_intermid += tmp2
    tmp1,tmp2 = MLPInfo(n_in+n_mid,n_mid,n_hidden,hidden_size)
    node += tmp1
    node_intermid += tmp2

    for _ in range(middle_layer):
        tmp1,tmp2 = MLPInfo(e_mid+2*n_mid+g_mid,e_mid,n_hidden,hidden_size)
        edge += tmp1
        edge_intermid += tmp2

        tmp1,tmp2 = MLPInfo(n_mid+e_mid+g_mid,n_mid,n_hidden,hidden_size)  
        node += tmp1
        node_intermid += tmp2
        tmp1,tmp2 = MLPInfo(n_mid+n_mid,n_mid,n_hidden,hidden_size)
        node += tmp1
        node_intermid += tmp2

    tmp1,tmp2 = MLPInfo(e_mid+2*n_mid+g_mid,e_out,n_hidden,hidden_size)
    edge += tmp1
    edge_intermid += tmp2

    tmp1,tmp2 = MLPInfo(n_mid+e_out+g_mid,n_out,n_hidden,hidden_size)  
    node += tmp1
    node_intermid += tmp2
    tmp1,tmp2 = MLPInfo(n_mid+n_out,n_out,n_hidden,hidden_size)
    node += tmp1
    node_intermid += tmp2

    total_para = edge + node 
    return total_para, edge_intermid, node_intermid

num_para,edge_intermid,node_intermid = GraphInfo()
mem = num_para * 8 /(10**6)
print('number of parameter:',num_para)
print(f'gpu memory of parameter is: {mem}M')

# csr = sparse.load_npz('../MatData/scipy_csr0.npz')
# csr = sparse.load_npz('../MatData/scipy_csr1.npz')
# csr = sparse.load_npz('../MatData/scipy_csr2.npz')
csr = sparse.load_npz('../MatData/scipy_csr3.npz')
nrow = csr.shape[0]
nnz = csr.nnz
print(f'nrow = {nrow}, nnz = {nnz}')

# total_mid = nrow*(node_intermid+1) + nnz*(edge_intermid+1)
total_mid = nrow*(node_intermid) + nnz*(edge_intermid)
print('node intermid layer:',node_intermid)
print('edge intermid layer:',edge_intermid)
print('number of intermid parameter:',total_mid)
print(f'gpu memory of intermid parameter is: {total_mid * 8/(10**6)}M')
print(f'gpu memory of the input matrix is: {(nrow*2+nnz) * 8/(10**6)}M')



