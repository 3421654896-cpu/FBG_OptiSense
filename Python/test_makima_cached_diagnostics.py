import unittest
from unittest.mock import patch
import numpy as np

import temporary_peak_fit_makima as m


def uncached(x,y):
    # Independent copy of the pre-cache numerical diagnostic, not the centre.
    origin=float(x[len(x)//2]);scale=float(np.median(np.diff(x)))
    u=(x-origin)/scale
    if float(np.ptp(y))<=1e-12:return 0.,0.
    centers,widths=np.meshgrid(np.linspace(float(u[0]+.25),float(u[-1]-.25),61),
                               np.linspace(.35,float(u[-1]-u[0]),32),indexing='ij')
    gaussian=np.exp(-.5*((u[None,:]-centers.ravel()[:,None])/widths.ravel()[:,None])**2)
    centered=gaussian-gaussian.mean(axis=1)[:,None]
    amplitude=(centered@(y-y.mean()))/np.maximum(np.sum(centered**2,axis=1),1e-12)
    baseline=y.mean()-amplitude*gaussian.mean(axis=1)
    negative=baseline<0;baseline[negative]=0
    amplitude[negative]=np.sum(gaussian[negative]*y,axis=1)/np.maximum(
        np.sum(gaussian[negative]**2,axis=1),1e-12)
    predicted=baseline[:,None]+amplitude[:,None]*gaussian
    sse=np.sum((predicted-y)**2,axis=1)
    sse[(amplitude<0)|~np.isfinite(sse)]=np.inf
    best=float(np.min(sse))
    if not np.isfinite(best):return 0.,float('inf')
    variance=float(np.sum((y-y.mean())**2))
    return 0. if variance<=1e-12 else float(1-best/variance),float(np.sqrt(best/len(x)))


class CacheTests(unittest.TestCase):
    def test_full_outputs_equal_legacy_diagnostic_for_5_7_regular_irregular_and_noisy(self):
        rng=np.random.default_rng(7)
        for count in (5,7):
            for irregular in (False,True):
                x=1530+np.arange(count)*.02
                if irregular:x[1:-1]+=rng.uniform(-.004,.004,count-2)
                for y in (np.zeros(count),np.full(count,40.),rng.uniform(0,4095,count),
                          100+1800*np.exp(-.5*((x-x[count//2])/.035)**2),
                          np.array([0.,*([2000.]*(count-2)),0.])):
                    actual=m.fit_peak_points(x,y)
                    with patch.object(m,'_gaussian_shape_diagnostics',uncached):
                        prior=m.fit_peak_points(x,y)
                    self.assertEqual(actual,prior)

    def test_cache_bounded_readonly_and_does_not_cache_adc_dependent_results(self):
        m._gaussian_shape_basis.cache_clear()
        x=1530+np.arange(7)*.02
        y=np.array([20,200,1000,2000,1000,200,20])
        first=m.fit_peak_points(x,y)
        second=m.fit_peak_points(x,y+np.arange(7)*30)
        self.assertNotEqual(first['center_nm'],second['center_nm'])
        self.assertGreater(m._gaussian_shape_basis.cache_info().hits,0)
        for basis in m._gaussian_shape_basis(tuple(x)):
            self.assertFalse(basis.flags.writeable)
        for index in range(80):m._gaussian_shape_basis(tuple(x+index*.0001))
        self.assertLessEqual(m._gaussian_shape_basis.cache_info().currsize,64)


if __name__=='__main__':
    unittest.main()
