import settings

from clustering import Clustering_Runner
from link_prediction import Link_pred_Runner


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
import argparse
import sys
parser = argparse.ArgumentParser()

parser.add_argument('--dataname')
parser.add_argument('--model')
parser.add_argument('--n_hop_enable')
parser.add_argument('--hop_count')

# fix conflict with tf.app.flags with addition -- separator
# python run.py -- --dataname=cora --model=arga_ae --n_hop_enable=True --hop_count=1
args = parser.parse_args(sys.argv[2:])
print('Arguments:', args)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

dataname = args.dataname   # 'cora' or 'citeseer' or 'pubmed'
model = args.model          # 'arga_ae' or 'arga_vae'
task = 'link_prediction'         # 'clustering' or 'link_prediction'
n_hop_enable = args.n_hop_enable                  # True or False                                                                                             # Author: Tonni
hop_count = args.hop_count                      # Degree of neighbours. For example, hop_count=1 means info till friends of neighbouring nodes              # Author: Tonni

settings = settings.get_settings(dataname, model, task, n_hop_enable, hop_count)
print("settings:", settings)

if task == 'clustering':
    runner = Clustering_Runner(settings)
else:
    runner = Link_pred_Runner(settings)

runner.erun()

