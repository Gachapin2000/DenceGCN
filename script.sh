IFS_BACKUP=$IFS
IFS=$'\n'

ary=('python3 train.py key=JKNet_Cora JKNet_Cora.pre_transform="HomophilyRank\(\)" JKNet_Cora.n_tri=100 JKNet_Cora.n_layer=8 JKNet_Cora.jk_mode=lstm JKNet_Cora.att_mode=mx JKNet_Cora.split=full_60per'
     'python3 train.py key=JKNet_CiteSeer JKNet_CiteSeer.pre_transform="HomophilyRank\(\)" JKNet_CiteSeer.n_tri=100 JKNet_CiteSeer.n_layer=1 JKNet_CiteSeer.jk_mode=lstm JKNet_CiteSeer.att_mode=go JKNet_CiteSeer.split=full_60per'
     'python3 train.py key=JKNet_CiteSeer JKNet_CiteSeer.pre_transform="HomophilyRank\(\)" JKNet_CiteSeer.n_tri=100 JKNet_CiteSeer.n_layer=2 JKNet_CiteSeer.jk_mode=lstm JKNet_CiteSeer.att_mode=go JKNet_CiteSeer.split=full_60per'
     'python3 train.py key=JKNet_CiteSeer JKNet_CiteSeer.pre_transform="HomophilyRank\(\)" JKNet_CiteSeer.n_tri=100 JKNet_CiteSeer.n_layer=3 JKNet_CiteSeer.jk_mode=lstm JKNet_CiteSeer.att_mode=go JKNet_CiteSeer.split=full_60per'
     'python3 train.py key=JKNet_CiteSeer JKNet_CiteSeer.pre_transform="HomophilyRank\(\)" JKNet_CiteSeer.n_tri=100 JKNet_CiteSeer.n_layer=5 JKNet_CiteSeer.jk_mode=lstm JKNet_CiteSeer.att_mode=go JKNet_CiteSeer.split=full_60per'
     'python3 train.py key=JKNet_CiteSeer JKNet_CiteSeer.pre_transform="HomophilyRank\(\)" JKNet_CiteSeer.n_tri=100 JKNet_CiteSeer.n_layer=8 JKNet_CiteSeer.jk_mode=lstm JKNet_CiteSeer.att_mode=go JKNet_CiteSeer.split=full_60per'
     'python3 train.py key=JKNet_CiteSeer JKNet_CiteSeer.pre_transform="HomophilyRank\(\)" JKNet_CiteSeer.n_tri=100 JKNet_CiteSeer.n_layer=1 JKNet_CiteSeer.jk_mode=lstm JKNet_CiteSeer.att_mode=mx JKNet_CiteSeer.split=full_60per'
     'python3 train.py key=JKNet_CiteSeer JKNet_CiteSeer.pre_transform="HomophilyRank\(\)" JKNet_CiteSeer.n_tri=100 JKNet_CiteSeer.n_layer=2 JKNet_CiteSeer.jk_mode=lstm JKNet_CiteSeer.att_mode=mx JKNet_CiteSeer.split=full_60per'
     'python3 train.py key=JKNet_CiteSeer JKNet_CiteSeer.pre_transform="HomophilyRank\(\)" JKNet_CiteSeer.n_tri=100 JKNet_CiteSeer.n_layer=3 JKNet_CiteSeer.jk_mode=lstm JKNet_CiteSeer.att_mode=mx JKNet_CiteSeer.split=full_60per'
     'python3 train.py key=JKNet_CiteSeer JKNet_CiteSeer.pre_transform="HomophilyRank\(\)" JKNet_CiteSeer.n_tri=100 JKNet_CiteSeer.n_layer=5 JKNet_CiteSeer.jk_mode=lstm JKNet_CiteSeer.att_mode=mx JKNet_CiteSeer.split=full_60per'
     'python3 train.py key=JKNet_CiteSeer JKNet_CiteSeer.pre_transform="HomophilyRank\(\)" JKNet_CiteSeer.n_tri=100 JKNet_CiteSeer.n_layer=8 JKNet_CiteSeer.jk_mode=lstm JKNet_CiteSeer.att_mode=mx JKNet_CiteSeer.split=full_60per')
for STR in ${ary[@]}
do
    echo "${STR}" >> "$1"
    eval "${STR}"
    echo '' >> "$1"
done >> "$1"