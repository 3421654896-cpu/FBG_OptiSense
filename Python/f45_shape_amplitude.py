"""Per-grating affine decomposition; retains amplitude and common responses."""
import numpy as np


def features(samples, baseline_samples):
    samples=np.asarray(samples,float); base=np.asarray(baseline_samples,float)
    if samples.ndim!=2 or samples.shape[1]!=90 or base.ndim!=2 or base.shape[1]!=90 or len(base)!=64:
        raise ValueError('Requires paired 90-code samples and 64-frame baseline')
    if not np.isfinite(samples).all() or not np.isfinite(base).all():
        raise ValueError('Nonfinite ADC')
    baseline=np.median(base,axis=0).reshape(18,5)
    noise=np.maximum(np.std(base,axis=0).reshape(18,5),1.)
    delta=samples.reshape(-1,18,5)-baseline
    centered=baseline-baseline.mean(axis=1,keepdims=True)
    # Fixed floor avoids amplification at flat/weak spectral segments.
    energy=np.maximum(np.sum(centered**2,axis=1),25.)
    gain_change=np.sum(delta*centered,axis=2)/energy
    mean_change=delta.mean(axis=2)
    residual=delta-mean_change[:,:,None]-gain_change[:,:,None]*centered
    # Do not subtract a common shift across the nine gratings. Amplitude terms
    # remain available because real pressure can change amplitude too.
    return np.concatenate([(residual/noise).reshape(len(samples),90),gain_change,
                           mean_change/np.sqrt(np.mean(noise**2,axis=1))],axis=1)
