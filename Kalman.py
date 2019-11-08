import numpy

class Kalman:
    def __init__(self, initial_state, initial_covar):
        self.last_state = initial_state
        self.last_covar = initial_covar

    def estimate(self, measured, acceleration, dt):
        A = numpy.array([
            [1, dt, 0, 0 ], # x
            [0,  1, 0, 0 ], # x'
            [0,  0, 1, dt], # y
            [0,  0, 0, 1 ]  #y'
        ])

        B = numpy.array([
            [.5 * dt ** 2, 0], #x''
            [dt,           0], #x'
            [0, .5 * dt ** 2], #y''
            [0, dt          ]  #y'
        ])

        state = numpy.dot(A, self.last_state) + numpy.dot(B, acceleration) # + noise

        self.last_state = state
        #self.last_covar = covar
        return state
