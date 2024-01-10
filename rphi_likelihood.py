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
    R, sigma_r, sigma_phi = params[:3]
    phases, xcents, ycents = np.split(params[3:], 3)
    #x,y = data

    npoints = np.sum([len(dt) for dt in data])
    prefact = -npoints*np.log(2*np.pi*sigma_phi*sigma_r)

    #k = np.arange(N)
    exp_likelihood = 0
    for i, sect in enumerate(data):
        x,y = sect
        # assume independent x,y
        xnorm = x - xcents[i]
        ynorm = y - ycents[i]

        r_data = np.sqrt(xnorm**2 + ynorm**2)

        segphases = 2*np.pi*np.arange(len(x))/N

        xi = R*np.cos(segphases + phases[i]) 
        yi = R*np.sin(segphases + phases[i])

        phase_data = np.arctan2(y, x)
        phase_model = np.arctan2(yi, xi)

        #xi = R*np.cos(2*np.pi*ks/N + phase) + xcent
        x_likelihood = -(r_data - R)**2/(2*sigma_r**2) - (phase_data - phase_model)**2/(2*sigma_phi**2)


        exp_likelihood += np.sum(x_likelihood)

   

    return prefact + exp_likelihood


def prior_bounds(nsegments):
    bounds = {
    "R": (60, 100),
    "sigma_r": (0,1),
    "sigma_phi": (0,1),
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

    plabels = ["R", "sigma_r", "sigma_phi"] + [f"phases{i}" for i in range(nsegments)] + [f"xcent{i}" for i in range(nsegments)] + [f"ycent{i}" for i in range(nsegments)]

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
    
    root_dir = "./rphi_likelihood2"
    data_path = "./1-Fragment_C_Hole_measurements.csv"

    run_nested(root_dir, data_path)
    plot_data.plot_all(root_dir)