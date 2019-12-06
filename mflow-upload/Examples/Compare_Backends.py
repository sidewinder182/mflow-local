import sys, os
sys.path.insert(0, os.path.abspath('..'))
os.chdir(os.path.abspath('..'))
import mkl
mkl.get_max_threads()
mkl.set_num_threads(1)

from Blocks.data_loader import extrasensory_data_loader
from Blocks.dummy_dataloader import dummy_data_loader
from Blocks.filter import MisingLabelFilter,  MisingDataColumnFilter, Take
from Blocks.imputer import Imputer
from Blocks.normalizer import Normalizer
from Blocks.experimental_protocol import ExpTrainTest, ExpCV, ExpWithin
from Blocks.results_analysis import ResultsConcat, ResultsCVSummarize, DataYieldReport

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.dummy import DummyClassifier

import matplotlib.pyplot as plt

from Workflow.workflow import workflow
import Workflow.compute_graph
import time
import pandas as pd

estimators = {"LR": LogisticRegression(solver="lbfgs",max_iter=1000)}
metrics    = [accuracy_score, f1_score, precision_score, recall_score]

res       = []
# df_raw    = extrasensory_data_loader(label="SLEEPING")
for i in range(1):
    df_raw    = dummy_data_loader()
    df_cf     = MisingDataColumnFilter(df_raw)
    df_lf     = MisingLabelFilter(df_cf)
    df_imp    = Imputer(df_lf)
    df_norm   = Normalizer(df_imp)
    res       += ExpTrainTest(df_norm, estimators, metrics=metrics)

configs = {
            "sequential":[1]
            # "multithread":[1,2,4]
            # "multiprocess":[1,2,4]
          }

results={}
for config in configs:
    for workers in configs[config]:
        
        print(config, workers)
        flow=workflow(res);        
        start = time.time()
        output=flow.run(backend=config, num_workers=workers, monitor=False, from_scratch=True);
        results[config+"(%d)"%(workers)] = time.time()-start
        print(config, workers, results[config+"(%d)"%(workers)])
    
time_df = pd.DataFrame(list(results.values()),columns=["Time(s)"], index=list(results.keys()))
print(time_df)
# display(time_df.plot(kind='bar', grid=True, title="Backend Runtime Comparison"))