IFS_BACKUP=$IFS
IFS=$'\n'

ary=('python3 train_reddit.py key=JKNet_Reddit JKNet_Reddit.n_layer=3 JKNet_Reddit.n_hid=256 JKNet_Reddit.epochs=20 JKNet_Reddit.n_tri=10 JKNet_Reddit.att_mode=go'
     'python3 train_reddit.py key=JKNet_Reddit JKNet_Reddit.n_layer=3 JKNet_Reddit.n_hid=256 JKNet_Reddit.epochs=20 JKNet_Reddit.n_tri=10 JKNet_Reddit.att_mode=mx'
     'python3 train_reddit.py key=JKNet_Reddit JKNet_Reddit.n_layer=3 JKNet_Reddit.n_hid=256 JKNet_Reddit.epochs=20 JKNet_Reddit.n_tri=10 JKNet_Reddit.att_mode=sd'
     'python3 train_reddit.py key=JKNet_Reddit JKNet_Reddit.n_layer=4 JKNet_Reddit.n_hid=256 JKNet_Reddit.epochs=20 JKNet_Reddit.n_tri=10 JKNet_Reddit.att_mode=go'
     'python3 train_reddit.py key=JKNet_Reddit JKNet_Reddit.n_layer=4 JKNet_Reddit.n_hid=256 JKNet_Reddit.epochs=20 JKNet_Reddit.n_tri=10 JKNet_Reddit.att_mode=mx'
     'python3 train_reddit.py key=JKNet_Reddit JKNet_Reddit.n_layer=4 JKNet_Reddit.n_hid=256 JKNet_Reddit.epochs=20 JKNet_Reddit.n_tri=10 JKNet_Reddit.att_mode=sd'
     'python3 train_reddit.py key=JKNet_Reddit JKNet_Reddit.n_layer=5 JKNet_Reddit.n_hid=256 JKNet_Reddit.epochs=20 JKNet_Reddit.n_tri=10 JKNet_Reddit.att_mode=go'
     'python3 train_reddit.py key=JKNet_Reddit JKNet_Reddit.n_layer=5 JKNet_Reddit.n_hid=256 JKNet_Reddit.epochs=20 JKNet_Reddit.n_tri=10 JKNet_Reddit.att_mode=mx'
     'python3 train_reddit.py key=JKNet_Reddit JKNet_Reddit.n_layer=5 JKNet_Reddit.n_hid=256 JKNet_Reddit.epochs=20 JKNet_Reddit.n_tri=10 JKNet_Reddit.att_mode=sd'
     'python3 train_reddit.py key=JKNet_Reddit JKNet_Reddit.n_layer=6 JKNet_Reddit.n_hid=256 JKNet_Reddit.epochs=20 JKNet_Reddit.n_tri=10 JKNet_Reddit.att_mode=go'
     'python3 train_reddit.py key=JKNet_Reddit JKNet_Reddit.n_layer=6 JKNet_Reddit.n_hid=256 JKNet_Reddit.epochs=20 JKNet_Reddit.n_tri=10 JKNet_Reddit.att_mode=mx'
     'python3 train_reddit.py key=JKNet_Reddit JKNet_Reddit.n_layer=6 JKNet_Reddit.n_hid=256 JKNet_Reddit.epochs=20 JKNet_Reddit.n_tri=10 JKNet_Reddit.att_mode=sd')

for STR in ${ary[@]}
do
    echo "${STR}" >> "$1"
    eval "${STR}"
    echo '' >> "$1"
done >> "$1"


# ----template----
# Cora     python3 train.py key=JKNet_Cora JKNet_Cora.n_tri=10 JKNet_Cora.n_layer=5 JKNet_Cora.jk_mode=lstm JKNet_Cora.att_mode=go JKNet_Cora.pre_transform="HomophilyRank\(\)"
# CiteSeer python3 train.py key=JKNet_CiteSeer JKNet_CiteSeer.n_tri=10 JKNet_CiteSeer.n_layer=5 JKNet_CiteSeer.jk_mode=lstm JKNet_CiteSeer.att_mode=go JKNet_CiteSeer.pre_transform="HomophilyRank\(\)"
# PubMed   python3 train.py key=JKNet_PubMed JKNet_PubMed.n_tri=10 JKNet_PubMed.n_layer=5 JKNet_PubMed.jk_mode=lstm JKNet_PubMed.att_mode=go JKNet_PubMed.pre_transform="HomophilyRank\(\)"
# PPI
# Reddit   python3 train_reddit.py key=JKNet_Reddit JKNet_Reddit.n_tri=100 JKNet_Reddit.n_layer=4 JKNet_Reddit.att_mode=sd