import unittest
import numpy as np
from kf import KF

class Testkf(unittest.TestCase):
    def test_can_construct_with_x_and_v(self):
        x = 0.2
        v = 0.5

        kf = KF(initial_x=x, initial_v=v, accel_vari=1.2)
        self.assertAlmostEqual(kf.pos, x)
        self.assertAlmostEqual(kf.vel, v)
        
    def test_after_calling_predict_X_and_P_are_correct_shape(self):
        x = 0.2
        v = 0.5
        dt = 0.1

        kf = KF(initial_x=x, initial_v=v, accel_vari=1.2)
        kf.predict(dt=dt)
        
        self.assertEqual(kf.cov.shape, (2,2))
        self.assertEqual(kf.mean.shape, (2, ))
        
    def test_after_calling_predict_X_increases_state_uncertanty(self):
        x = 0.2
        v = 0.5
        dt = 0.1

        kf = KF(initial_x=x, initial_v=v, accel_vari=1.2)
        
        for i in range (10):
            det_before = np.linalg.det(kf.cov)
            kf.predict(dt=dt)
            det_after = np.linalg.det(kf.cov)
        
        self.assertGreater(det_after,det_before)
        print(det_before,det_after)
    
    def test_calling_update_does_not_crash(self):
        x = 0.2
        v = 0.5
        
        kf = KF(initial_x=x, initial_v=v, accel_vari=1.2)
        kf.update(meas_value=0.1, meas_variance=0.1)
