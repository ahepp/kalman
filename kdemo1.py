#!/usr/bin/python3

import numpy

from Kalman import Kalman

# based on https://www.youtube.com/watch?v=fwM66qohrJ8

initial_state = numpy.array([
    [4000], # x
    [280], # x'
    [3000],# y
    [120]  # y'
])
initial_covar = numpy.array([
    [400, 0, 0, 0], # x
    [0, 25, 0, 0], # x'
    [0, 0, 0, 0], # y
    [0, 0, 0, 0]  # y'
])

acceleration = numpy.array([
    [2],
    [0]
])

dt = 1

kfilter = Kalman(initial_state, initial_covar)

observation = numpy.array([
    [4260], # x
    [282],  # x'
    [0],    # y
    [0]     # y'
])
kfilter.estimate(observation, acceleration, dt)

observation = numpy.array([
    [4550], # x
    [285],  # x'
    [0],    # y
    [0]     # y'
])

kfilter.estimate(observation, acceleration, dt)

