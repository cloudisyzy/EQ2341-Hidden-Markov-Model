import numpy as np

def compute_pX(observations, dists, scale=True):
    ## to estimate bj(x) in textbook, observations <=> x
    ##                                dists <=> distribution used to generate x
    ##                                likelihood <=> bj(x) for each distribution
    unscaled_pX = []

    for dist in dists:
        likelihood = dist.prob(observations)
        unscaled_pX.append(likelihood)

    unscaled_pX =  np.array(unscaled_pX)
    scaled_pX = unscaled_pX / np.max(unscaled_pX, axis=0) # scale as described in textbook

    if scale == True:
        return scaled_pX
    else:
        return unscaled_pX