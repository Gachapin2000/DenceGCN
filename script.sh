IFS_BACKUP=$IFS
IFS=$'\n'

ary=('python3 train.py key=JKNet_Cora JKNet_Cora.n_hid=16 JKNet_Cora.jk_mode=lstm +JKNet_Cora.att_mode=go'
     'python3 train.py key=JKNet_Cora JKNet_Cora.n_hid=16 JKNet_Cora.jk_mode=lstm +JKNet_Cora.att_mode=sd'
     'python3 train.py key=JKNet_Cora JKNet_Cora.n_hid=16 JKNet_Cora.jk_mode=lstm +JKNet_Cora.att_mode=mx'
     'python3 train.py key=JKNet_CiteSeer JKNet_CiteSeer.n_hid=16 JKNet_CiteSeer.jk_mode=lstm +JKNet_CiteSeer.att_mode=go'
     'python3 train.py key=JKNet_CiteSeer JKNet_CiteSeer.n_hid=16 JKNet_CiteSeer.jk_mode=lstm +JKNet_CiteSeer.att_mode=sd'
     'python3 train.py key=JKNet_CiteSeer JKNet_CiteSeer.n_hid=16 JKNet_CiteSeer.jk_mode=lstm +JKNet_CiteSeer.att_mode=mx')

for STR in ${ary[@]}
do
    echo "${STR}" >> "$1"
    eval "${STR}"
    echo '' >> "$1"
done >> "$1"