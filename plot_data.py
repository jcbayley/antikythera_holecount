import numpy as np
import matplotlib.pyplot as plt
import os
import load_data
from dynesty import NestedSampler
from dynesty import plotting as dyplot

def plot_circle_points(root_dir, data):

    fig, ax = plt.subplots()
    xc,yc = 80, 135
    r = 77
    xcirc = r*np.cos(np.linspace(0,2*np.pi)) + xc
    ycirc = r*np.sin(np.linspace(0,2*np.pi)) + yc
    ax.plot(xc,yc, "o")
    ax.plot(xcirc,ycirc, "k",alpha=0.5)
    for i,dt in enumerate(data):
        ax.plot(dt[0], dt[1], marker="o", ms=2,ls="none",label = f"Fragment {i}")

    ax.set_xlabel("Xposition")
    ax.set_ylabel("Yposition")
    ax.set_xlim([0, 120])
    ax.set_ylim([50, 150])
    ax.legend(loc="upper right")

    fig.savefig(os.path.join(root_dir, "circle_points.png"))

def plot_marginal_posteriors(root_dir):

    with open(os.path.join(root_dir,"parnames.txt"), "r") as f:
        plabels = f.readlines()

    Ns = []
    for fname in sorted(os.listdir(root_dir)):
        if fname.startswith("dynesty"):
            Ns.append(int(fname.split("_")[1].split(".")[0]))

    for n in Ns:
        fsampler = NestedSampler.restore(os.path.join(root_dir,f'dynesty_{n}.save'))
        fres = fsampler.results

        fg, ax = dyplot.cornerplot(fres, color='C0', labels=plabels, dims=[0,1,2],
                            show_titles=True, quantiles=None,)
        
        fg.savefig(os.path.join(root_dir, f"marg_posterior_{n}.png"))

def plot_evidences(root_dir):

    Ns, log_evidences = load_data.load_evidences(root_dir)

    fig, ax = plt.subplots()
    sind,eind = 0,len(Ns)
    ax.plot(Ns[sind:eind],log_evidences[sind:eind])
    ax.set_xlabel("N")
    ax.set_ylabel("log p(d | N)")
    fig.savefig(os.path.join(root_dir, "evidences.png"))



def plot_all(root_dir):

    plot_evidences(root_dir)

    plot_marginal_posteriors(root_dir)