import numpy

class Kalman:
    def __init__(self, initial_state, initial_covar):
        self.last_state = initial_state
        self.last_covar = initial_covar

    def estimate(self, measurement, acceleration, dt):
        # Predict the state and covariance
        A = numpy.array([
            [1, dt, 0, 0 ], # x
            [0,  1, 0, 0 ], # x'
            [0,  0, 1, dt], # y
            [0,  0, 0, 1 ]  # y'
        ])
        a = .5 * dt ** 2 # acceleration term
        B = numpy.array([
            [a,  0 ], #x''
            [dt, 0 ], #x'
            [0,  dt], #y''
            [0,  a ]  #y'
        ])
        predicted_state = numpy.dot(A, self.last_state) + numpy.dot(B, acceleration) # + noise
        predicted_covar = numpy.dot(numpy.dot(A, self.last_covar), numpy.transpose(A)) # + noise

        # Mask the covariance, since x and y are known to be independent
        predicted_covar = numpy.diag(numpy.diag(predicted_covar))


        # Extract position to incorporate into our model
        C = numpy.array([
            [1, 0, 0, 0], # x
            [0, 1, 0, 0], # x'
            [0, 0, 0, 0], # y
            [0, 0, 0, 0]  # y'
        ])
        observation = numpy.dot(C, measurement) # + noise

        # Calculate Kalman gain to correct our prediction based on our measurement
        I = numpy.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        R = numpy.array([ # Observation covariance
            [625, 0,  0, 0],
            [0,   36, 0, 0],
            [0,   0,  0, 0],
            [0,   0,  0, 0]
        ])
        with numpy.errstate(divide='ignore', invalid='ignore'):
            kalman_gain = predicted_covar / (predicted_covar + R) # Kalman gain represents confidence in measurement over prediction
        kalman_gain = numpy.nan_to_num(kalman_gain)

        print('Kalman gain:')
        print(kalman_gain)


        corrected_state = predicted_state + numpy.dot(kalman_gain, observation - predicted_state)
        corrected_covar = numpy.dot(I - kalman_gain, predicted_covar)

        print('Result:')
        print(corrected_state)

        self.last_state = corrected_state
        self.last_covar = corrected_covar
        return corrected_state


