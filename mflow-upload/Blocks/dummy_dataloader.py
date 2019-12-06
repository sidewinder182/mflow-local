from Workflow.compute_graph import node

import pandas as pd
import time
import numpy as np


def dummy_data_loader(**kwargs):
    return node(function = __dummy_data_loader, kwargs=kwargs, name=" Dummy Data Loader")

def __dummy_data_loader():
    col_list = []
    for i in range(200):
        col_list.append(str(i))
    print(len(col_list))

    df = pd.DataFrame(np.random.normal(0,100,size=(2000000, 200)), columns=col_list)
    # df=df.rename(index=str, columns={"199": "target"})
    df2 = pd.DataFrame(np.random.randint(0,2,size=(2000000,1)), columns=['target'])
    df = pd.concat([df, df2], axis=1, sort=False)
    # df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
    df.index.names = ["ID"]

    return({"dataframe":df})