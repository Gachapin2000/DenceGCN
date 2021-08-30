IFS_BACKUP=$IFS
IFS=$'\n'

ary=("python3 train_ppi.py -m 'hydra.sweeper.n_trials=10' 'key=AttGNN_PPI' 'mlflow.runname=AttGNN_PPI' 'AttGNN_PPI.n_tri=5' 'AttGNN_PPI.epochs=200' 'AttGNN_PPI.learning_rate=0.001' 'AttGNN_PPI.weight_decay=0.' 'AttGNN_PPI.dropout=0.' 'AttGNN_PPI.n_hid=128' 'AttGNN_PPI.n_layer=2' 'AttGNN_PPI.summary_mode=choice(lstm,vanilla,roll)' 'AttGNN_PPI.att_mode=choice(ad,dp,mx)' 'AttGNN_PPI.att_temparature=1.' 'AttGNN_PPI.learn_temparature=False'
      python3 train_ppi.py -m 'hydra.sweeper.n_trials=10' 'key=AttGNN_PPI' 'mlflow.runname=AttGNN_PPI' 'AttGNN_PPI.n_tri=5' 'AttGNN_PPI.epochs=200' 'AttGNN_PPI.learning_rate=0.001' 'AttGNN_PPI.weight_decay=0.' 'AttGNN_PPI.dropout=0.' 'AttGNN_PPI.n_hid=128' 'AttGNN_PPI.n_layer=3' 'AttGNN_PPI.summary_mode=choice(lstm,vanilla,roll)' 'AttGNN_PPI.att_mode=choice(ad,dp,mx)' 'AttGNN_PPI.att_temparature=1.' 'AttGNN_PPI.learn_temparature=False'
      python3 train_ppi.py -m 'hydra.sweeper.n_trials=10' 'key=AttGNN_PPI' 'mlflow.runname=AttGNN_PPI' 'AttGNN_PPI.n_tri=5' 'AttGNN_PPI.epochs=200' 'AttGNN_PPI.learning_rate=0.001' 'AttGNN_PPI.weight_decay=0.' 'AttGNN_PPI.dropout=0.' 'AttGNN_PPI.n_hid=128' 'AttGNN_PPI.n_layer=4' 'AttGNN_PPI.summary_mode=choice(lstm,vanilla,roll)' 'AttGNN_PPI.att_mode=choice(ad,dp,mx)' 'AttGNN_PPI.att_temparature=1.' 'AttGNN_PPI.learn_temparature=False'
      python3 train_ppi.py -m 'hydra.sweeper.n_trials=10' 'key=AttGNN_PPI' 'mlflow.runname=AttGNN_PPI' 'AttGNN_PPI.n_tri=5' 'AttGNN_PPI.epochs=200' 'AttGNN_PPI.learning_rate=0.001' 'AttGNN_PPI.weight_decay=0.' 'AttGNN_PPI.dropout=0.' 'AttGNN_PPI.n_hid=128' 'AttGNN_PPI.n_layer=5' 'AttGNN_PPI.summary_mode=choice(lstm,vanilla,roll)' 'AttGNN_PPI.att_mode=choice(ad,dp,mx)' 'AttGNN_PPI.att_temparature=1.' 'AttGNN_PPI.learn_temparature=False'
      python3 train_ppi.py -m 'hydra.sweeper.n_trials=10' 'key=AttGNN_PPI' 'mlflow.runname=AttGNN_PPI' 'AttGNN_PPI.n_tri=5' 'AttGNN_PPI.epochs=200' 'AttGNN_PPI.learning_rate=0.001' 'AttGNN_PPI.weight_decay=0.' 'AttGNN_PPI.dropout=0.' 'AttGNN_PPI.n_hid=128' 'AttGNN_PPI.n_layer=6' 'AttGNN_PPI.summary_mode=choice(lstm,vanilla,roll)' 'AttGNN_PPI.att_mode=choice(ad,dp,mx)' 'AttGNN_PPI.att_temparature=1.' 'AttGNN_PPI.learn_temparature=False'")

for STR in ${ary[@]}
do
    eval "${STR}"
done


# ----template----
# Cora     python3 train.py -m 'hydra.sweeper.n_trials=500' 'key=AttGNN_Cora' 'mlflow.runname=AttGNN_Cora' 'AttGNN_Cora.learning_rate=choice(0.01,0.005,0.001)' 'AttGNN_Cora.weight_decay=choice(0.001,0.0005,0.0001,0.)' 'AttGNN_Cora.dropout=choice(0.6,0.5,0.)' 'AttGNN_Cora.n_layer=range(2,6)' 'AttGNN_Cora.att_mode=choice('ad','mx','dp')' 'AttGNN_Cora.att_temparature=interval(-1.,1.)'
# CiteSeer python3 train.py -m 'hydra.sweeper.n_trials=500' 'key=AttGNN_CiteSeer' 'mlflow.runname=AttGNN_CiteSeer' 'AttGNN_CiteSeer.learning_rate=choice(0.01,0.005,0.001)' 'AttGNN_Cora.weight_decay=choice(0.001,0.0005,0.0001,0.)' 'AttGNN_CiteSeer.dropout=choice(0.6,0.5,0.)' 'AttGNN_CiteSeer.n_layer=range(2,6)' 'AttGNN_CiteSeer.att_mode=choice('ad','mx','dp')' 'AttGNN_CiteSeer.att_temparature=interval(-1.,1.)'
# PubMed   python3 train.py -m 'hydra.sweeper.n_trials=100' 'key=AttGNN_PubMed' 'mlflow.runname=AttGNN_PubMed' 'AttGNN_PubMed.learning_rate=choice(0.01,0.005,0.001)' 'AttGNN_PubMed.weight_decay=choice(0.001,0.0005,0.0001,0.)' 'AttGNN_PubMed.dropout=choice(0.6,0.5,0.)' 'AttGNN_PubMed.n_layer=range(2,6)' 'AttGNN_PubMed.att_mode=choice('ad','mx','dp')' 'AttGNN_PubMed.att_temparature=interval(-1.,1.)'
# PPI      python3 train_ppi.py -m 'hydra.sweeper.n_trials=100' 'key=AttGNN_PPI' 'mlflow.runname=AttGNN_PPI' 'AttGNN_PPI.learning_rate=choice(0.01,0.005,0.001)' 'AttGNN_PPI.weight_decay=choice(0.001,0.0005,0.0001,0.)' 'AttGNN_PPI.dropout=choice(0.6,0.5,0.)' 'AttGNN_PPI.n_layer=range(2,6)' 'AttGNN_PPI.att_mode=choice('ad','mx','dp')' 'AttGNN_PPI.att_temparature=interval(-1.,1.)'
# Reddit   python3 train_reddit.py -m 'hydra.sweeper.n_trials=50' 'key=AttGNN_Reddit' 'mlflow.runname=AttGNN_Reddit_summarizer' 'AttGNN_Reddit.n_tri=5' 'AttGNN_Reddit.epochs=20' 'AttGNN_Reddit.learning_rate=0.001' 'AttGNN_Reddit.weight_decay=0.' 'AttGNN_Reddit.dropout=0.' 'AttGNN_Reddit.n_layer=2' 'AttGNN_Reddit.n_hid=128' 'AttGNN_Reddit.summary_mode=choice('vanilla', 'roll', 'lstm')' 'AttGNN_Reddit.att_mode=choice('ad','mx','dp')' 'AttGNN_Reddit.att_temparature=interval(-1.,1.)'

