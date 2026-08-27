attractors.txt: [55 x 4 x 15000] array of timeseries data. The first dimension is each of 55 different experimental trajectories, the second dimension is 4d state space (y, theta, dy, dtheta), and the third dimension is time.

The units of length are in meters, and the units of time are in frames (all data sampled at 3000fps).

ex: attractors(1, 1, :) is y(t) of the first experimental dataset and  attractors(1, 2, :) is theta(t) of the first experimental dataset

y0: [55x1] array of fitted y0, which is the constant offset needed to have y(t) be in a gravity free frame


---------------

run_AGATE_weights.m: MATLAB function to run the AGATE method on data stored in attractors.txt, outputs optimized coefficients for minimal model specified in function



