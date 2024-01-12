import pandas 
import numpy as np
import os
from dynesty import NestedSampler
from dynesty import plotting as dyplot

def load_antikythera(fname):
    data = pandas.read_csv(fname)

    data_col = data[["Section ID", "Mean(X)", "Mean(Y)"]]

    data_groupeddf = data_col.groupby(data_col["Section ID"])[["Mean(X)", "Mean(Y)"]].agg(lambda x: x.tolist()).values

    data_grouped = []
    for xt, yt in data_groupeddf:
        xn = np.array(xt)
        yn = np.array(yt)
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