import numpy as np

# def compute_pX(observations, dists, scale=True):
#     ## to estimate bj(x) in textbook, observations <=> x
#     ##                                dists <=> distribution used to generate x
#     ##                                likelihood <=> bj(x) for each distribution
#     unscaled_pX = []

#     for dist in dists:
#         pX_i = []
#         for i in range(observations.shape[1]):
#             likelihood = dist.prob(observations[:,i])
#             pX_i.append(likelihood)
#         unscaled_pX.append(pX_i)

#     unscaled_pX =  np.array(unscaled_pX)
#     scaled_pX = unscaled_pX / np.max(unscaled_pX, axis=0) # scale as described in textbook

#     if scale == True:
#         return scaled_pX
#     else:
#         return unscaled_pX

def compute_pX(observations, dists, scale=True):
    ## to estimate bj(x) in textbook, observations <=> x
    ##                                dists <=> distribution used to generate x
    ##                                likelihood <=> bj(x) for each distribution
    range_dim = observations.shape[1] if observations.ndim > 1 else observations.shape[0]
    unscaled_pX = []

    for dist in dists:
        pX_i = []
        for i in range(range_dim):
            if observations.ndim > 1:
                likelihood = dist.prob(observations[:,i])
            else:
                likelihood = dist.prob(observations[i])
            pX_i.append(likelihood)
        unscaled_pX.append(pX_i)

    unscaled_pX =  np.array(unscaled_pX)
    scaled_pX = unscaled_pX / np.max(unscaled_pX, axis=0) # scale as described in textbook

    if scale == True:
        return scaled_pX
    else:
        return unscaled_pX