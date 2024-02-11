import numpy as np

# offsets of each variable in the state vector
iX = 0
iV = 1
NUMVARS = iV + 1


class KF:
    def __init__(self, initial_x: float, 
                       initial_v: float,
                       accel_vari: float) -> None:
        # mean of state GRV
        self._X = np.array([initial_x, initial_v])
        self._accelVari = accel_vari
                
        # initial of state GRV
        self._P = np.eye(2)
        
    def predict(self, dt: float) -> None:
        # x = F*x
        # P = F * P * Ft + G* Gt * a
        F = np.array([[1, dt], [0,1]])
        G = np.array([0.5*dt**2, dt]).reshape([2,1])
        
        new_X = F.dot(self._X)
        
        new_P = F.dot(self._P).dot(F.T) + G.dot(G.T)*self._accelVari
        
        self._P = new_P
        self._X = new_X
        
    def update(self, meas_value: float, meas_variance: float):
        # y = z - H x
        # S = H P Ht + R
        # K = P Ht S^-1
        # X = x + K y
        # p = (i - k H) * P
        
        H = np.array([1, 0]).reshape([1,2])
        
        Z = np.array([meas_value])
        R = np.array([meas_variance])
        
        y = Z - H.dot(self._X)
        S = H.dot(self._P).dot(H.T) + R
        
        K = self._P.dot(H.T).dot(np.linalg.inv(S))

        new_x = self._X + K.dot(y)
        new_P = (np.eye(2) - K.dot(H)).dot(self._P)
        
        self._X = new_x
        self._P = new_P
    
    @property
    def cov(self) -> np.array:
        return self._P
    
    @property
    def mean(self) -> np.array:
        return self._X
    
    @property
    def pos(self) -> float:
        return self._X[0]

    @property
    def vel(self) -> float:
        return self._X[1]