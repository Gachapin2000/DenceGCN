IFS_BACKUP=$IFS
IFS=$'\n'

ary=('python3 train.py key=JKNet_Cora JKNet_Cora.n_layer=5 JKNet_Cora.jk_mode=lstm JKNet_Cora.att_mode=go'
     'python3 train.py key=JKNet_Cora JKNet_Cora.n_layer=5 JKNet_Cora.jk_mode=lstm JKNet_Cora.att_mode=mx'
     'python3 train.py key=JKNet_Cora JKNet_Cora.n_layer=5 JKNet_Cora.jk_mode=lstm JKNet_Cora.att_mode=sd')

for STR in ${ary[@]}s
do
    echo "${STR}" >> "$1"
    eval "${STR}"
    echo '' >> "$1"
done >> "$1"