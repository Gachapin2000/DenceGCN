IFS_BACKUP=$IFS
IFS=$'\n'

ary=("python3 train_ppi.py -m 'hydra.sweeper.n_trials=50' 'key=JKNet_PPI' 'mlflow.runname=JKNet_PPI_l3' 'JKNet_PPI.learning_rate=choice(0.001,0.005,0.01)' 'JKNet_PPI.weight_decay=choice(0.,0.0001,0.0005,0.001)' 'JKNet_PPI.dropout=choice(0.,0.5,0.6)' 'JKNet_PPI.n_layer=3' 'JKNet_PPI.n_hid=choice(128,256)' 'JKNet_PPI.att_mode=choice('ad','mx','dp')' 'JKNet_PPI.att_temparature=interval(-0.6,0.6)'
      python3 train_ppi.py -m 'hydra.sweeper.n_trials=50' 'key=JKNet_PPI' 'mlflow.runname=JKNet_PPI_l4' 'JKNet_PPI.learning_rate=choice(0.001,0.005,0.01)' 'JKNet_PPI.weight_decay=choice(0.,0.0001,0.0005,0.001)' 'JKNet_PPI.dropout=choice(0.,0.5,0.6)' 'JKNet_PPI.n_layer=4' 'JKNet_PPI.n_hid=choice(128,256)' 'JKNet_PPI.att_mode=choice('ad','mx','dp')' 'JKNet_PPI.att_temparature=interval(-0.6,0.6)'
      python3 train_ppi.py -m 'hydra.sweeper.n_trials=50' 'key=JKNet_PPI' 'mlflow.runname=JKNet_PPI_l5' 'JKNet_PPI.learning_rate=choice(0.001,0.005,0.01)' 'JKNet_PPI.weight_decay=choice(0.,0.0001,0.0005,0.001)' 'JKNet_PPI.dropout=choice(0.,0.5,0.6)' 'JKNet_PPI.n_layer=5' 'JKNet_PPI.n_hid=choice(128,256)' 'JKNet_PPI.att_mode=choice('ad','mx','dp')' 'JKNet_PPI.att_temparature=interval(-0.6,0.6)'
      python3 train_ppi.py -m 'hydra.sweeper.n_trials=50' 'key=JKNet_PPI' 'mlflow.runname=JKNet_PPI_l6' 'JKNet_PPI.learning_rate=choice(0.001,0.005,0.01)' 'JKNet_PPI.weight_decay=choice(0.,0.0001,0.0005,0.001)' 'JKNet_PPI.dropout=choice(0.,0.5,0.6)' 'JKNet_PPI.n_layer=6' 'JKNet_PPI.n_hid=choice(128,256)' 'JKNet_PPI.att_mode=choice('ad','mx','dp')' 'JKNet_PPI.att_temparature=interval(-0.6,0.6)'")

for STR in ${ary[@]}
do
    echo "${STR}" >> "$1"
    eval "${STR}"
    echo '' >> "$1"
done >> "$1"


# ----template----
# Cora     python3 train.py -m 'hydra.sweeper.n_trials=500' 'key=JKNet_Cora' 'mlflow.runname=JKNet_Cora' 'JKNet_Cora.learning_rate=choice(0.01,0.005,0.001)' 'JKNet_Cora.weight_decay=choice(0.001,0.0005,0.0001,0.)' 'JKNet_Cora.dropout=choice(0.6,0.5,0.)' 'JKNet_Cora.n_layer=range(2,6)' 'JKNet_Cora.att_mode=choice('ad','mx','dp')' 'JKNet_Cora.att_temparature=interval(-1.,1.)'
# CiteSeer python3 train.py -m 'hydra.sweeper.n_trials=500' 'key=JKNet_CiteSeer' 'mlflow.runname=JKNet_CiteSeer' 'JKNet_CiteSeer.learning_rate=choice(0.01,0.005,0.001)' 'JKNet_Cora.weight_decay=choice(0.001,0.0005,0.0001,0.)' 'JKNet_CiteSeer.dropout=choice(0.6,0.5,0.)' 'JKNet_CiteSeer.n_layer=range(2,6)' 'JKNet_CiteSeer.att_mode=choice('ad','mx','dp')' 'JKNet_CiteSeer.att_temparature=interval(-1.,1.)'
# PubMed   python3 train.py -m 'hydra.sweeper.n_trials=100' 'key=JKNet_PubMed' 'mlflow.runname=JKNet_PubMed' 'JKNet_PubMed.learning_rate=choice(0.01,0.005,0.001)' 'JKNet_PubMed.weight_decay=choice(0.001,0.0005,0.0001,0.)' 'JKNet_PubMed.dropout=choice(0.6,0.5,0.)' 'JKNet_PubMed.n_layer=range(2,6)' 'JKNet_PubMed.att_mode=choice('ad','mx','dp')' 'JKNet_PubMed.att_temparature=interval(-1.,1.)'
# PPI      python3 train_ppi.py -m 'hydra.sweeper.n_trials=100' 'key=JKNet_PPI' 'mlflow.runname=JKNet_PPI' 'JKNet_PPI.learning_rate=choice(0.01,0.005,0.001)' 'JKNet_PPI.weight_decay=choice(0.001,0.0005,0.0001,0.)' 'JKNet_PPI.dropout=choice(0.6,0.5,0.)' 'JKNet_PPI.n_layer=range(2,6)' 'JKNet_PPI.att_mode=choice('ad','mx','dp')' 'JKNet_PPI.att_temparature=interval(-1.,1.)'
# Reddit   python3 train_reddit.py -m 'hydra.sweeper.n_trials=100' 'key=JKNet_Reddit' 'mlflow.runname=JKNet_Reddit' 'JKNet_Reddit.learning_rate=choice(0.01,0.005,0.001)' 'JKNet_Reddit.weight_decay=choice(0.001,0.0005,0.0001,0.)' 'JKNet_Reddit.dropout=choice(0.6,0.5,0.)' 'JKNet_Reddit.n_layer=range(2,6)' 'JKNet_Reddit.att_mode=choice('ad','mx','dp')' 'JKNet_Reddit.att_temparature=interval(-1.,1.)'
