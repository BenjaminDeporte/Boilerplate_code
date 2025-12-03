import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')   # headless render (no GUI)
import matplotlib.pyplot as plt
import seaborn as sns
import torch as torch
import torch.nn as nn
from tqdm import tqdm


class MHSampler():
    """ 
    Metropolis-Hastings Sampler
    """
    default_scale = 1e-2
    
    def __init__(self, target_dist, d=1, scale=None, burn_in=10000, interval=50):
        """
        Create the class with the following parameters:
        - target_dist: callable, computes the unnormalized target density at a given point x
            x is a d-dimensional np.array
        - scale : standard deviation of the normal distribution that is used by defaults as the proposal distribution. Defaults to 0.1
        - burn_in : int, number of initial samples to discard, in order to allow the Markov chain to converge to the target distribution
        - samples : int, number of samples to generate after burn-in
        - interval : int, number of steps between recorded samples to reduce autocorrelation
        """
        self.dim = d  # dimension of the data
        self.target_dist = target_dist
        self.interval = interval
        
        if scale is None:
            self.scale = np.eye(d) * self.default_scale
        else:
            self.scale = np.eye(d) * scale
        
        self.burn_in = burn_in
        self.interval = interval
        self.current_sample = None
        
    def __repr__(self):
        return f"MHSampler(target_dist={self.target_dist}, burn_in={self.burn_in}, interval={self.interval})"
    
    def _sample(self):
        """
        Draw one sample from the chain
        """
        if self.current_sample is None:
            self.current_sample = np.zeros(self.dim)
        z_current = self.current_sample
        
        proposal_distribution = torch.distributions.MultivariateNormal(
            loc=torch.tensor(self.current_sample), 
            covariance_matrix=torch.tensor(self.scale)
        )
        z_proposal = proposal_distribution.rsample().numpy()  # from tensor to numpy float
        
        # NB : the proposal distribution is Normal, therefore symetric, so the acceptance criterion reduces to Metropolis
        acceptance_ratio = min(1 , (self.target_dist(z_proposal) / self.target_dist(z_current)))
        if np.random.rand() < acceptance_ratio:
            accepted = True
            z_next = z_proposal
        else:
            accepted = False
            z_next = z_current
        self.current_sample = z_next
        return z_next, accepted
    
    def run(self, samples):
        """
        Run the MH algorithm to generate samples from the target distribution, starting from an initial sample.
        """
        self.current_sample = np.zeros(self.dim)
        total_accepted = 0
        
        print(f'Burn-in')
        for _ in tqdm(range(self.burn_in)):
            _, accepted = self._sample()
            total_accepted += accepted
        print(f'Burn-in : acceptance is {total_accepted/self.burn_in*100:.2f}%')
            
        print(f'Sampling')
        total_accepted = 0
        list_samples = []
        for _ in tqdm(range(samples)):
            s, acc = self._sample()
            list_samples.append(s)
            total_accepted += acc
            for _ in range(self.interval):
                _,acc = self._sample()
                total_accepted += acc
        print(f'Sampling : {total_accepted} accepted, ie {total_accepted/(samples*self.interval)*100:.2f}%')
                
        return list_samples
    
# number of samples
K = 1000

# target distribution : gaussian in dim 2
xc = 1.0
yc = 2.0
mu = np.array([xc,yc])# build cov matrix
theta = np.pi / 6
s1 = 0.05
s2 = 2.00
diag = np.array([[s1,0],[0,s2]])
orth = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta), np.cos(theta)]])
scale = orth.T @ diag @ orth
inv_scale = np.linalg.inv(scale)

# plot from torch.distributions
torch_dist = torch.distributions.MultivariateNormal(
    loc=torch.tensor(mu),
    covariance_matrix=torch.tensor(scale)
)
gts = torch_dist.rsample((K,)).numpy()

# sampling from custom class
def target_dist(x):
    return np.exp(-1/2 * (x-mu).T @ inv_scale @ (x-mu))

# NB : the scale of the proposal distribution should be the same order of magnitude as sigma_min in case of a Gaussian multivariate
mh = MHSampler(target_dist=target_dist, scale=s1, d=2)
xs = np.array(mh.run(samples=K))

# plots
x_max = np.max([np.max(gts[:,0]), np.max(xs[:,0])])
x_min = np.min([np.min(gts[:,0]), np.min(xs[:,0])])
y_max = np.max([np.max(gts[:,1]), np.max(xs[:,1])])
y_min = np.min([np.min(gts[:,1]), np.min(xs[:,1])])
eps = 0.1

fig, ax = plt.subplots(figsize=(21,6),nrows=1, ncols=3)
ax[0].scatter(gts[:,0], gts[:,1], marker='.')
ax[0].set_title(f'Sampling from\n torch.distribution.MultivariateNormal')
ax[0].scatter(xc,yc,marker='x',color='red')
ax[0].grid()
ax[0].set_xlim(x_min-eps, x_max+eps)
ax[0].set_ylim(y_min-eps, y_max+eps)
ax[1].scatter(xs[:,0], xs[:,1], marker='.')
ax[1].set_title(f'Sampling from\n Metropolis Hastings home-made Class')
ax[1].scatter(xc,yc,marker='x',color='red')
ax[1].grid()
ax[1].set_xlim(x_min-eps, x_max+eps)
ax[1].set_ylim(y_min-eps, y_max+eps)
ax[2].scatter(gts[:,0], gts[:,1], marker='.', alpha=0.3, color='blue', label='Torch')
ax[2].scatter(xs[:,0], xs[:,1], marker='.', alpha=0.3, color='green', label='MH Class')
ax[2].scatter(xc,yc,marker='x',color='red')
ax[2].grid()
ax[2].set_xlim(x_min-eps, x_max+eps)
ax[2].set_ylim(y_min-eps, y_max+eps)
ax[2].legend()
ax[2].set_title(f'Both')

fig.suptitle(f'Samplings')
plt.show()