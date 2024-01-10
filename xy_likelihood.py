import numpy as np
from dynesty import NestedSampler
from dynesty import plotting as dyplot
import dill
import dynesty.utils
dynesty.utils.pickle_module = dill
import load_data
import plot_data
import os

def model(R, phase, xcent, ycent):

    xi = R*np.cos(phase) + xcent
    yi = R*np.sin(phase) + ycent

    return xi, yi

def log_likelihood(params, data, N):
    R, sigma_x, sigma_y = params[:3]
    phases, xcents, ycents = np.split(params[3:], 3)
    #x,y = data

    #k = np.arange(N)
    total_likelihood = 0
    for i, sect in enumerate(data):
        x,y = sect
        ks = np.arange(len(x))

        phase = 2*np.pi*ks/N + phases[i]
        # assume independent x,y
        xi, yi = model(R, phase, xcents[i], ycents[i])

        #xi = R*np.cos(2*np.pi*ks/N + phase) + xcent
        x_likelihood = -(xi - x)**2/(2*sigma_x**2)

        #yi = R*np.sin(2*np.pi*ks/N + phase) + ycent
        y_likelihood = -(yi - y)**2/(2*sigma_y**2)

        normlogfact = -len(x)/2. * np.log(2*np.pi*sigma_x*sigma_y)

        total_likelihood += 2*normlogfact + np.sum(x_likelihood + y_likelihood)

    return total_likelihood


def prior_bounds(nsegments):
    bounds = {
    "R": (60, 100),
    "sigma_x": (0,1),
    "sigma_y": (0,1),
}
    for k in range(nsegments):
        bounds[f"phases{k}"] = (0,2*np.pi)
    for k in range(nsegments):
        bounds[f"xcent{k}"] = (70,90)
    for k in range(nsegments):
        bounds[f"ycent{k}"] = (130,140)

    return bounds

def prior_transform(u, bounds, plabels):
    r, phase, sigma = u[:3]
    ks = u[3:]

    outvals = []
    for i,key in enumerate(plabels):
        outvals.append(u[i]*(bounds[key][1] - bounds[key][0]) + bounds[key][0])

    return outvals


def run_nested(root_dir, data_path):

    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)

    data = load_data.load_antikythera(data_path)
    nsegments = len(data)

    bounds = prior_bounds(nsegments)

    plabels = ["R", "sigma_x", "sigma_y"] + [f"phases{i}" for i in range(nsegments)] + [f"xcent{i}" for i in range(nsegments)] + [f"ycent{i}" for i in range(nsegments)]

    with open(os.path.join(root_dir,"parnames.txt"), "w") as f:
        for line in plabels:
            f.write(f"{line}\n")

    anti_logzs = []
    anti_samples = []
    ndims = 3 + 3*nsegments
    Nrange = np.arange(352, 367)#np.array([353, 354, 355, 359, 360, 361])

    
    print(ndims)
    for n in Nrange:
        andyll = lambda params: log_likelihood(params, data, n)
        andypt = lambda params: prior_transform(params, bounds, plabels)

        sampler = NestedSampler(andyll, andypt, ndim=ndims, nlive=500)

        sampler.run_nested(checkpoint_file=os.path.join(root_dir, f'dynesty_{n}.save'))

        res = sampler.results

        anti_logzs.append(res.logz[-1])

if __name__ == "__main__":
    
    root_dir = "./xy_likelihood2"
    data_path = "./1-Fragment_C_Hole_measurements.csv"


    run_nested(root_dir, data_path)
    plot_data.plot_all(root_dir)