import numpy as np
from dynesty import NestedSampler
from dynesty import plotting as dyplot
import dill
import dynesty.utils
dynesty.utils.pickle_module = dill
import load_data
import plot_data
import os

def model(x, y, R, phis, xcent, ycent, phase):

    # find phase for each hole in model
    phi = phis + phase

    cphi = np.cos(phi)
    sphi = np.sin(phi)

    # compute model points in x,y
    r_x = R*cphi
    r_y = R*sphi

    # shift data point to be around model x,y
    d_x = x - xcent
    d_y = y - ycent

    # find error vector between data and model
    e_x = r_x - d_x
    e_y = r_y - d_y

    # project vector into radius and tangent
    rp = e_x*cphi + e_y*sphi
    tp = e_x*sphi - e_y*cphi

    return rp, tp

def log_likelihood(params, data, N):
    R, sigma_r, sigma_t = params[:3]
    phases, xcents, ycents = np.split(params[3:], 3)
    #x,y = data

    invsig_r = 1./(2*(sigma_r*sigma_r))
    invsig_t = 1./(2*(sigma_t*sigma_t))

    #npoints = np.sum([len(dt) for dt in data])
    prefact = 0#-npoints*np.log(2*np.pi*sigma_t*sigma_r)

    phis = 2*np.pi*np.arange(100)/N

    #k = np.arange(N)
    exp_likelihood = 0
    for i, sect in enumerate(data):
        x,y = sect

        # assume independent r, tangent
        rp, tp = model(x, y, R, phis[:len(x)], xcents[i], ycents[i], phases[i])

        exponent = -invsig_r*(rp**2) - invsig_t*(tp**2)

        prefact_i = -len(x)*np.log(2*np.pi*sigma_t*sigma_r)

        exp_likelihood += np.sum(exponent) + prefact_i


    return prefact + exp_likelihood


def log_likelihood_wrong(params, data, N):
    R, sigma_r, sigma_t = params[:3]
    phases, xcents, ycents = np.split(params[3:], 3)
    #x,y = data

    invsig_r = 1./(2*(sigma_r*sigma_r))
    invsig_t = 1./(2*(sigma_t*sigma_t))

    npoints = np.sum([len(dt) for dt in data])
    prefact = 0#-npoints*np.log(2*np.pi*sigma_t*sigma_r)
    phis = 2*np.pi*np.arange(100)/N

    #k = np.arange(N)
    exp_likelihood = 0
    for i, sect in enumerate(data):
        x,y = sect

        # assume independent r, tangent
        rp, tp = model(x, y, R, phis[:len(x)], xcents[i], ycents[i], phases[i])

        exponent = -invsig_r*(rp**2) - invsig_t*(tp**2)

        prefact_i = -len(x)*np.log(2*np.pi*sigma_t*sigma_r)

        exp_likelihood += np.sum(prefact_i + exponent)


    return prefact + exp_likelihood


def prior_bounds(nsegments):
    bounds = {
        "R": (60, 100),
        "sigma_r": (0,1),
        "sigma_t": (0,1),
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


def run_nested(root_dir, data_path, wrong_likelihood = False, segments=None, remove_endpoints=False, remove_singles=False):

    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)

    data = load_data.load_antikythera(data_path, segments=segments, remove_endpoints=remove_endpoints, remove_singles=remove_singles)
    nsegments = len(data)

    bounds = prior_bounds(nsegments)

    plabels = ["R", "sigma_r", "sigma_t"] + [f"phases{i}" for i in range(nsegments)] + [f"xcent{i}" for i in range(nsegments)] + [f"ycent{i}" for i in range(nsegments)]

    with open(os.path.join(root_dir,"parnames.txt"), "w") as f:
        for line in plabels:
            f.write(f"{line}\n")

    anti_logzs = []

    ndims = 3 + 3*nsegments
    Nrange = np.arange(350, 367) # np.array([353, 354, 355, 359, 360, 361])

    for n in Nrange:
        if wrong_likelihood:
            andyll = lambda params: log_likelihood_wrong(params, data, n)
        else:
            andyll = lambda params: log_likelihood(params, data, n)

        andypt = lambda params: prior_transform(params, bounds, plabels)

        sampler = NestedSampler(andyll, andypt, ndim=ndims, nlive=4000)

        sampler.run_nested(checkpoint_file=os.path.join(root_dir, f'dynesty_{n}.save'), dlogz=0.1)

        res = sampler.results

        anti_logzs.append(res.logz[-1])

if __name__ == "__main__":

    #segments = [1,2,3,5,6,7]
    segments = [1,2,3,7]

    remove_endpoints = True

    if segments is not None:
        seg_str = ""
        for i in segments:
            seg_str += str(i)
    else:
        seg_str = "none"

    root_dir = f"./dotprod_dynesty_4000live_{seg_str}_{remove_endpoints}_remove_singles"

    data_path = "./1-Fragment_C_Hole_measurements.csv"


    run_nested(
        root_dir, 
        data_path, 
        wrong_likelihood=False,
        segments=segments,
        remove_singles=True,
        remove_endpoints=remove_endpoints)
    
    plot_data.plot_all(root_dir)