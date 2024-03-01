import numpy as np
from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.livepoint import live_points_to_dict
from nessai.utils import setup_logger
import load_data
import plot_data
import os
import csv
from dotprod_likelihood_dynesty_nsample import log_likelihood

class GaussianModel(Model):
    """A simple two-dimensional Gaussian likelihood."""

    def __init__(self, data):
        nsegments = len(data)
        self.names = ["N", "R", "sigma_r", "sigma_t"] + [f"phases{i}" for i in range(nsegments)] + [f"xcent{i}" for i in range(nsegments)] + [f"ycent{i}" for i in range(nsegments)]

        # Prior bounds for each parameter
        self.bounds = self.prior_bounds(nsegments)

        self.data = data

    def prior_bounds(self, nsegments):
        bounds = {
            "N": (340, 370),
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

    def log_prior(self, x):
        """
        Returns log of prior given a live point assuming uniform
        priors on each parameter.
        """
        # Check if values are in bounds, returns True/False
        # Then take the log to get 0/-inf and make sure the dtype is float
        log_p = np.log(self.in_bounds(x), dtype="float")
        # Iterate through each parameter (x and y)
        # since the live points are a structured array we can
        # get each value using just the name
        for n in self.names:
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        return log_p
    
    def hole_model(self, x, y, R, phis, xcent, ycent, phase):

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

    def log_likelihood(self, x):
        """
        Returns log likelihood of given live point assuming a Gaussian
        likelihood.
        """

        x = live_points_to_dict(x, self.names)

        params = np.array([x[parname] for parname in self.names])
        exp_likelihood = log_likelihood(params, self.data)


        #R, sigma_r, sigma_t = x[0][:3]
        #phases, xcents, ycents = np.split(x[0][3:], 3)
        #x,y = data
        """
        invsig_r = 1./(2*(x["sigma_r"]*x["sigma_r"]))
        invsig_t = 1./(2*(x["sigma_t"]*x["sigma_t"]))

        phis = 2*np.pi*np.arange(100)/x["N"]

        #k = np.arange(N)
        exp_likelihood = 0
        for i, sect in enumerate(self.data):
            xpos,ypos = sect

            # assume independent r, tangent
            rp, tp = self.hole_model(xpos, ypos, x["R"], phis[:len(xpos)], x[f"xcent{i}"], x[f"ycent{i}"], x[f"phases{i}"])

            exponent = -invsig_r*(rp**2) - invsig_t*(tp**2)

            # X HERE SHOULD BE XPOS len(x) -> len(xpos)
            prefact_i = -len(x)*np.log(2*np.pi*x["sigma_t"]*x["sigma_r"])

            exp_likelihood += np.sum(exponent) + prefact_i

        """
        return exp_likelihood




def run_nested(root_dir, data_path, segments=None, remove_endpoints=False, remove_singles=True):


    used_segments, data = load_data.load_antikythera(data_path, segments=segments, remove_endpoints=remove_endpoints, remove_singles=remove_singles)
    nsegments = len(data)

    print(data)

    seg_str = ""
    for i in used_segments:
        seg_str += str(i)

    sub_dir = f"./dotprod_nsample_nessai_{seg_str}"
    if remove_endpoints:
        sub_dir += "_remove_endpoints"
    if remove_singles:
        sub_dir += "_remove_singles"

    root_dir = os.path.join(root_dir, sub_dir)
    
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)

    with open(os.path.join(root_dir, "data_used.csv"), "w") as f:
        wr = csv.writer(f)
        wr.writerows(data)

    model = GaussianModel(data)


    with open(os.path.join(root_dir,"parnames.txt"), "w") as f:
        for line in model.names:
            f.write(f"{line}\n")

    anti_logzs = []

    ndims = 3 + 3*nsegments

    output = os.path.join(root_dir, f"nessai")
    logger = setup_logger(output=output)
        
    sampler = FlowSampler(model, output=output, nlive=2000)
    sampler.run()

if __name__ == "__main__":

    #segments = [1,2,3,5,6,7]
    segments = None
    segments = [1,2,3,7]

    remove_endpoints = True
    remove_singles = True


    root_dir = f"./dotprod_nsample_nessai_2"

    data_path = "./1-Fragment_C_Hole_measurements.csv"

    run_nested(root_dir, data_path, segments=segments, remove_endpoints=remove_endpoints, remove_singles=remove_singles)
