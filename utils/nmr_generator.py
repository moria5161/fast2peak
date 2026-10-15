import h5py
import numpy as np
import math
import os
import time
from multiprocessing import Pool, cpu_count
from functools import partial

# ==========================================
# Configuration.
# ==========================================
CONFIG = {
    'simulation': {
        'num_samples': 10000,      # Total number of samples.
        'length': 16000,            # Spectrum length.
        'ppm_range': (0, 50),       # NMR ppm range.
    },
    'noise': {
        'type': 'gaussian',
        'gaussian_range': (1e-4, 3e-3), # Gaussian noise standard deviation range.
    },
    'system': {
        'num_cores': 24,            # Number of worker processes.
        'save_path': '/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/5161-zhangnannan/fast2peak/datasets/nmr/train.h5'
    }
}

class NMRSpectraGenerator:
    def pseudo_voigt(self, position, fwhm, length):
        # Generate a narrow Pseudo-Voigt peak with a strong Lorentzian component.
        x = np.linspace(0, length - 1, length)
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        gaussian = (1 / (sigma * np.sqrt(2 * math.pi)) * np.exp(-(x - position) ** 2 / (2 * sigma**2)))
        lorentzian = (1 / math.pi) * (fwhm / 2) / ((x - position) ** 2 + (fwhm / 2) ** 2)
        
        alpha = 0.85 # Lorentzian weight for long NMR peak tails.
        pv_curve = (1 - alpha) * gaussian + alpha * lorentzian
        
        pv_min, pv_max = np.min(pv_curve), np.max(pv_curve)
        if pv_max - pv_min == 0: return pv_curve
        return (pv_curve - pv_min) / (pv_max - pv_min)

    def generate_dataset(self, num_samples, length, cfg_noise):
        spec_gt = np.zeros((num_samples, length))
        
        for idx in range(num_samples):
            n_peaks = np.random.randint(10, 21) # 10-20 peaks.
            
            # Keep peak centers away from the edges.
            pos_arr = np.random.randint(200, length - 200, size=n_peaks)
            # Narrow FWHM: 3-15 points in a 16000-point spectrum.
            fwhm_arr = np.random.uniform(3, 15, size=n_peaks) 
            # Use an exponential distribution for many weak peaks and a few strong peaks.
            int_arr = np.random.exponential(scale=20, size=n_peaks) + 2 

            for i in range(n_peaks):
                spec_gt[idx, :] += self.pseudo_voigt(pos_arr[i], fwhm_arr[i], length) * int_arr[i]

        # Normalize GT.
        max_vals = np.max(spec_gt, axis=1, keepdims=True)
        max_vals[max_vals == 0] = 1 # Avoid division by zero.
        spec_gt = spec_gt / max_vals

        # Add Gaussian noise only.
        levels = np.random.uniform(*cfg_noise['gaussian_range'], size=num_samples)
        noise = np.random.normal(0, 1, spec_gt.shape) * levels[:, np.newaxis]
        spec_noisy = spec_gt + noise

        return spec_gt, spec_noisy

def worker_task(chunk_size, length, cfg_noise):
    np.random.seed() # Use independent random seeds in worker processes.
    gen = NMRSpectraGenerator()
    return gen.generate_dataset(chunk_size, length, cfg_noise)

def main():
    print("Generating simulated NMR spectra")
    t0 = time.time()
    
    total_samples = CONFIG['simulation']['num_samples']
    length = CONFIG['simulation']['length']
    ppm = np.linspace(*CONFIG['simulation']['ppm_range'], length)
    n_cores = min(CONFIG['system']['num_cores'], cpu_count())
    
    # Assign work to workers.
    chunk_size = total_samples // n_cores
    remainder = total_samples % n_cores
    tasks = [chunk_size + 1 if i < remainder else chunk_size for i in range(n_cores)]
    
    worker_func = partial(worker_task, length=length, cfg_noise=CONFIG['noise'])
    
    with Pool(n_cores) as pool:
        results = pool.map(worker_func, tasks)

    print("Combining generated data...")
    gt_list, noisy_list = zip(*results)
    
    # 100000 spectra of length 16000 require substantial RAM (about 25 GB).
    data_gt = np.concatenate(gt_list, axis=0)
    data_noisy = np.concatenate(noisy_list, axis=0)

    save_path = CONFIG['system']['save_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"Writing H5 file: {save_path}")
    with h5py.File(save_path, 'w') as f:
        # Store the ppm axis under wavenumbers for compatibility.
        f.create_dataset('noisy', data=data_noisy, compression="gzip", compression_opts=4, chunks=True)
        f.create_dataset('gt', data=data_gt, compression="gzip", compression_opts=4, chunks=True)
        f.create_dataset('wavenumbers', data=ppm, compression="gzip", compression_opts=4, chunks=True)

    print(f"Completed in {time.time() - t0:.2f} s")

if __name__ == "__main__":
    main()
