import settings

from clustering import Clustering_Runner
from link_prediction import Link_pred_Runner


dataname = 'cora'       # 'cora' or 'citeseer' or 'pubmed'
model = 'arga_ae'          # 'arga_ae' or 'arga_vae'
task = 'link_prediction'         # 'clustering' or 'link_prediction'
n_hop_enable = False                  # True or False                                                                                             # Author: Tonni
hop_count = 0                       # Degree of neighbours. For example, hop_count=1 means info till friends of neighbouring nodes              # Author: Tonni

settings = settings.get_settings(dataname, model, task, n_hop_enable, hop_count)
print("settings:", settings)

if task == 'clustering':
    runner = Clustering_Runner(settings)
else:
    runner = Link_pred_Runner(settings)

runner.erun()

