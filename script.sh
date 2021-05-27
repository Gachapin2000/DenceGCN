IFS_BACKUP=$IFS
IFS=$'\n'

ary=('python3 train_ppi.py key=JKNet_PPI JKNet_PPI.n_tri=100 JKNet_PPI.att_mode=go JKNet_PPI.n_layer=2'
     'python3 train_ppi.py key=JKNet_PPI JKNet_PPI.n_tri=100 JKNet_PPI.att_mode=go JKNet_PPI.n_layer=3'
     'python3 train_ppi.py key=JKNet_PPI JKNet_PPI.n_tri=100 JKNet_PPI.att_mode=go JKNet_PPI.n_layer=4'
     'python3 train_ppi.py key=JKNet_PPI JKNet_PPI.n_tri=100 JKNet_PPI.att_mode=go JKNet_PPI.n_layer=5'
     'python3 train_ppi.py key=JKNet_PPI JKNet_PPI.n_tri=100 JKNet_PPI.att_mode=mx JKNet_PPI.n_layer=2'
     'python3 train_ppi.py key=JKNet_PPI JKNet_PPI.n_tri=100 JKNet_PPI.att_mode=mx JKNet_PPI.n_layer=3'
     'python3 train_ppi.py key=JKNet_PPI JKNet_PPI.n_tri=100 JKNet_PPI.att_mode=mx JKNet_PPI.n_layer=4'
     'python3 train_ppi.py key=JKNet_PPI JKNet_PPI.n_tri=100 JKNet_PPI.att_mode=mx JKNet_PPI.n_layer=5'
     'python3 train_ppi.py key=JKNet_PPI JKNet_PPI.n_tri=100 JKNet_PPI.att_mode=sd JKNet_PPI.n_layer=2'
     'python3 train_ppi.py key=JKNet_PPI JKNet_PPI.n_tri=100 JKNet_PPI.att_mode=sd JKNet_PPI.n_layer=3'
     'python3 train_ppi.py key=JKNet_PPI JKNet_PPI.n_tri=100 JKNet_PPI.att_mode=sd JKNet_PPI.n_layer=4'
     'python3 train_ppi.py key=JKNet_PPI JKNet_PPI.n_tri=100 JKNet_PPI.att_mode=sd JKNet_PPI.n_layer=5'
     'python3 train_ppi.py key=JKNet_PPI JKNet_PPI.n_tri=100 JKNet_PPI.att_mode=sd+ JKNet_PPI.n_layer=2'
     'python3 train_ppi.py key=JKNet_PPI JKNet_PPI.n_tri=100 JKNet_PPI.att_mode=sd+ JKNet_PPI.n_layer=3'
     'python3 train_ppi.py key=JKNet_PPI JKNet_PPI.n_tri=100 JKNet_PPI.att_mode=sd+ JKNet_PPI.n_layer=4'
     'python3 train_ppi.py key=JKNet_PPI JKNet_PPI.n_tri=100 JKNet_PPI.att_mode=sd+ JKNet_PPI.n_layer=5')

for STR in ${ary[@]}
do
    echo "${STR}" >> "$1"
    eval "${STR}"
    echo '' >> "$1"
done >> "$1"