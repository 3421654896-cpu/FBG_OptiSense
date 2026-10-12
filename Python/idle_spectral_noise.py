"""Offline noise metric learned only from an independent optical idle capture."""
import json
import numpy as np
from analyze_current_idle_warmup import analyze
from evaluate_current_three_positions import spectral_feature


def whitening_matrix(samples):
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] < 100 or samples.shape[1] != 45 or not np.isfinite(samples).all():
        raise ValueError('At least 100 finite 45-dimensional idle vectors required')
    covariance = np.cov(samples, rowvar=False)
    # Fixed 50% spherical shrinkage, chosen before inspecting query results.
    scale = max(float(np.trace(covariance) / 45), 1e-8)
    covariance = .5 * covariance + .5 * scale * np.eye(45)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return (eigenvectors * (1 / np.sqrt(eigenvalues))) @ eigenvectors.T


def load_idle_metric(path):
    audit = analyze(path)  # Completeness, no CNC commands, sequence and shutdown gates.
    if audit['duration_s'] < 100:
        raise ValueError('Noise capture must cover at least 100 seconds')
    data = json.loads(path.read_text('utf-8'))
    values = np.array([[r['first_code'] for r in f['records']]
                       for f in data['frames'] if f['sequence'] >= 16], dtype=float)
    baseline = np.median(values, axis=0)
    features = [spectral_feature(baseline, row, 'shape_delta') for row in values]
    return whitening_matrix(features), dict(source=audit['source'], sha256=audit['sha256'],
        shrinkage=.5, training_source='Independent idle spectrum only; no contact labels',
        firmware_table_sha256=data['table_sha256'], table_crc32=data['table_codes_crc32'])
