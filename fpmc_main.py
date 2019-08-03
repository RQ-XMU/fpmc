import time
import numpy as np
import pandas as pd
import adapter as ad
import evaluation_test_last as eval_last
from metrics import accuracy as ac

if __name__ == '__main__':
    export_csv = 'results/fpmc.csv'
    metric = []
    metric.append(ac.HitRate(20))
    metric.append(ac.HitRate(15))
    metric.append(ac.HitRate(10))
    metric.append(ac.HitRate(5))
    metric.append(ac.MRR(20))
    metric.append(ac.MRR(15))
    metric.append(ac.MRR(10))
    metric.append(ac.MRR(5))

    # train_df = pd.read_csv("DataSet/yelp.train.rating", sep='\t', header=None, usecols=[0, 1, 2, 3], dtype={0: np.int32, 1: np.int64, 2: np.float32, 3: str})
    # train_df.columns = ["SessionId", "ItemId", "Rating", "Time"]
    # train_df.sort_values(["SessionId", "Time"], inplace=True)
    # train_df = train_df.reset_index(drop=True)
    #
    # test_df = pd.read_csv("DataSet/yelp.test.rating", sep='\t', header=None, usecols=[0, 1, 2, 3],
    #                        dtype={0: np.int32, 1: np.int64, 2: np.float32, 3: str})
    # test_df.columns = ["SessionId", "ItemId", "Rating", "Time"]
    # test_df.sort_values(["SessionId", "Time"], inplace=True)
    # test_df = test_df.reset_index(drop=True)
    # full_data = pd.read_csv("DataSet/yelp.train.rating", sep='\t', header=None, usecols=[0, 1, 2, 3], dtype={0: np.int32, 1: np.int64, 2: np.float32, 3: str})
    # full_data.columns = ["SessionId", "ItemId", "Rating", "Time"]
    # full_data.sort_values(["SessionId", "Time"], inplace=True)
    # full_data = full_data.reset_index(drop=True)
    # list_train = []
    # list_test = []
    # for userid in range(full_data["SessionId"].nunique()):
    #     list_train.append(full_data[full_data.SessionId == userid][:-2])
    #     list_test.append(full_data[full_data.SessionId == userid][-2:])
    #
    # train_df = pd.concat(list_train, axis=0)
    # test_df = pd.concat(list_test, axis=0)
    # train_df = train_df.reset_index(drop=True)
    # test_df = test_df.reset_index(drop=True)
    # del (train_df['Rating'])
    # del (test_df['Rating'])
    pr = ad.Adapter()

    for m in metric:
        m.init(pr.instance.trainset)

    ts = time.time()
    print("  fit  ", "fpmc")
    pr.fit(pr.instance.trainset)

    res = {}

    res["fpmc"] = eval_last.evaluate_sessions(pr, metric, pr.instance.testset, pr.instance.valset)

    for k, l in res.items():
        for e in l:
            print(k, ":", e[0], " ", e[1])

    if export_csv is not None:
        file = open(export_csv, 'w+')
        file.write('Metrics:')

        for k, l in res.items():
            for e in l:
                file.write(e[0])
                file.write(';')
            break
        file.write('\n')

        for k, l in res.items():
            file.write(k)
            file.write(";")
            for e in l:
                file.write(str(e[1]))
                file.write(';')
            file.write('\n')