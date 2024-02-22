import pandas 
import numpy as np
import os
from dynesty import NestedSampler
from dynesty import plotting as dyplot

def load_antikythera(fname, segments = None, remove_endpoints=False, remove_singles=True):
    data = pandas.read_csv(fname)

    data_col = data[["Section ID", "Mean(X)", "Mean(Y)"]]

    data_groupeddf = data_col.groupby(data_col["Section ID"])[["Mean(X)", "Mean(Y)"]].agg(lambda x: x.tolist()).values

    if segments is None:
        segments = np.arange(len(data_groupeddf))

    data_grouped = []
    for i, (xt, yt) in enumerate(data_groupeddf):
        if i not in segments:
            continue
        if remove_endpoints:
            xn = np.array(xt)[1:-1]
            yn = np.array(yt)[1:-1]
        else:
            xn = np.array(xt)
            yn = np.array(yt)
        if remove_singles:
            print(i, len(xn), len(yn), len(xt), len(yt))
            if len(xn) <= 1:
                continue
        data_grouped.append([xn, yn])

    return data_grouped


def load_evidences(root_dir):
    Ns = []
    log_evidences = []
    for fname in sorted(os.listdir(root_dir)):
        if fname.startswith("dynesty"):
            n = int(fname.split("_")[1].split(".")[0])
            Ns.append(n)

            t_sampler = NestedSampler.restore(os.path.join(root_dir, fname))
            t_res = t_sampler.results

            log_evidences.append(t_res.logz[-1])

    return Ns, log_evidences