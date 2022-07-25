"""
Markov Chain Monte Carlo sampling control logic.
"""
import scipy.stats as stats
from fulgurite.tree import PhyloTree


def sample(tree: PhyloTree, samples, weight=0.2):
    """Metropolis algorithm.
    Estimate Mk model rate parameter Q by constructing an ergodic, reversible
    Markov chain, the stationary distribution of which is the same as the
    posterior that we're trying to estimate. It's kind of like magic tbh.
    """
    posterior = []
    likelihood = []
    qcurrent = stats.uniform(0, 1).rvs()
    for i in range(samples):
        # Select proposed value of q based on proposal weight density
        qproposal = stats.uniform(qcurrent - weight / 2, qcurrent + weight / 2).rvs()

        # Calculate prior odds ratio: the ratio of the prior probability of the
        # proposal divided by the prior probability of the current value. And
        # to get the prior probability of each value we just need to call the
        # probability density function of the prior distribution with that value.
        prior_ratio = stats.uniform.pdf(qproposal) / stats.uniform.pdf(qcurrent)
        
        # Calculate likelihoods
        lcurrent = tree.root.get_likelihood(qcurrent)
        lproposal = tree.root.get_likelihood(qproposal)
        likely_ratio = lproposal / lcurrent

        # Calculate acceptance ratio and decide whether to accept proposal
        accept_ratio = likely_ratio * prior_ratio
        if stats.uniform(0, 1).rvs() < accept_ratio:
            qcurrent = qproposal

        # Add this sample to the set of posterior samples
        posterior.append(qcurrent)
        likelihood.append(lcurrent)

    return posterior, likelihood
