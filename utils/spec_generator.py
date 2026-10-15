import scipy.io as sio
import numpy as np
import random
import math
import os
import time
from multiprocessing import Pool, cpu_count
from functools import partial

# ==========================================
# 1. Configuration.
# ==========================================
CONFIG = {
    'simulation': {
        'num_samples': 200000,        # Total number of samples.
        'length': 1600,             # Spectrum length.
        'wn_range': (0, 1000),      # Wavenumber range.
    },
    'physics': {
        'add_rayleigh': False,      # Add Rayleigh background.
        'add_fluorescence': False,  # Add fluorescence background.
    },
    'noise': {
        'type': 'mixed',            # Noise type: 'gaussian', 'poisson', or 'mixed'.
        'gaussian_range': (1e-4, 1e-1),
        'poisson_range': (3e-1, 2e-1),
    },
    'system': {
        'num_cores': 24,            # Number of worker processes.
        'save_dir': 'datasets/simu'
    }
}

# ==========================================
# 2. Spectrum generator.
# ==========================================
class SpectraGenerator:
    def __init__(self):
        pass

    def pseudo_voigt(self, position, fwhm, length):
        # Generate a Pseudo-Voigt peak.
        x = np.linspace(0, length - 1, length)
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        gaussian = (1 / (sigma * np.sqrt(2 * math.pi)) * np.exp(-(x - position) ** 2 / (2 * sigma**2)))
        lorentzian = (1 / math.pi) * (fwhm / 2) / ((x - position) ** 2 + (fwhm / 2) ** 2)
        alpha = 0.6785 
        pv_curve = (1 - alpha) * gaussian + alpha * lorentzian
        
        # Normalize.
        pv_min, pv_max = np.min(pv_curve), np.max(pv_curve)
        if pv_max - pv_min == 0: return pv_curve
        return (pv_curve - pv_min) / (pv_max - pv_min)

    def generate_random_spectra(self, length, batch_size):
        # Generate clean peaks and record their parameters.
        spectra = np.zeros((batch_size, length))
        
        # Parameter records.
        recs = {'n_peaks': [], 'pos': [], 'fwhm': [], 'int': []}

        for idx in range(batch_size):
            n_peaks = np.random.randint(3, 21) 
            
            # Generate parameters in batches.
            pos_arr = np.random.randint(200, length - 200, size=n_peaks)
            fwhm_arr = np.random.randint(3, 50, size=n_peaks)
            int_arr = np.random.randint(5, 50, size=n_peaks)

            # Sum the peak profiles.
            for i in range(n_peaks):
                spectra[idx, :] += self.pseudo_voigt(pos_arr[i], fwhm_arr[i], length) * int_arr[i]

            # Record peak parameters.
            recs['n_peaks'].append(n_peaks)
            recs['pos'].extend(pos_arr)
            recs['fwhm'].extend(fwhm_arr)
            recs['int'].extend(int_arr)

        return (spectra, np.array(recs['n_peaks']), np.array(recs['pos']), 
                np.array(recs['fwhm']), np.array(recs['int']))

    # Background generation.
    def add_Rayleigh_background(self, spec, wavenumbers):
        batch_size = spec.shape[0]
        x = wavenumbers.reshape(1, -1)
        amp = np.random.uniform(1, 10, batch_size).reshape(-1, 1)
        gamma = np.random.uniform(50, 200, batch_size).reshape(-1, 1)
        bg = amp * (gamma**2) / (x**2 + gamma**2)
        return spec + bg, bg

    def add_Fluorescence_background(self, spec, wavenumbers):
        batch_size = spec.shape[0]
        x = wavenumbers.reshape(1, -1)
        amp = np.random.uniform(0.5, 10, batch_size).reshape(-1, 1)
        center = np.random.uniform(200, 800, batch_size).reshape(-1, 1)
        sigma = np.random.uniform(300, 600, batch_size).reshape(-1, 1)
        bg = amp * np.exp(-0.5 * ((x - center) / sigma)**2)
        return spec + bg, bg

    # Noise generation.
    def add_GaussianNoise(self, spectra, rng):
        levels = np.random.uniform(*rng, size=spectra.shape[0])
        noise = np.random.normal(0, 1, spectra.shape) * levels[:, np.newaxis]
        return spectra + noise, levels

    def add_PoissonNoise(self, spectra, rng):
        levels = np.random.uniform(*rng, size=(spectra.shape[0], 1)) * 1e5
        lam = np.maximum(spectra, 1e-6) * levels
        # Keep lam nonnegative for numerical stability.
        noise = np.random.poisson(lam) / levels - spectra
        return spectra + noise, levels

    def generate_dataset(self, num_samples, length, wavenumbers, cfg_phys, cfg_noise):
        # Generate a batch of paired spectra.
        # 1. Generate clean spectra.
        (spec_gt, r_num, r_pos, r_fwhm, r_int) = self.generate_random_spectra(length, num_samples)
        
        # Normalize GT.
        spec_gt = spec_gt / np.max(spec_gt, axis=1, keepdims=True)
        spec_peaks = spec_gt.copy()
        

        bg_total = np.zeros_like(spec_peaks)
        if cfg_phys['add_rayleigh']:
            _, bg_r = self.add_Rayleigh_background(spec_peaks, wavenumbers)
            bg_total += bg_r
            
        if cfg_phys['add_fluorescence']:
            _, bg_f = self.add_Fluorescence_background(spec_peaks, wavenumbers)
            bg_total += bg_f

        spec_gt_raw = spec_peaks + bg_total
        spec_gt = spec_gt_raw / np.max(spec_gt_raw, axis=1, keepdims=True)
        spec_noisy_raw = spec_gt.copy()

        ntype = cfg_noise['type']
        
        # Initialize noise levels to None.
        gaussian_levels = None
        poisson_levels = None
        
        if ntype == 'gaussian':
            spec_noisy_raw, gaussian_levels = self.add_GaussianNoise(spec_noisy_raw, cfg_noise['gaussian_range'])
            
        elif ntype == 'poisson':
            spec_noisy_raw, poisson_levels = self.add_PoissonNoise(spec_noisy_raw, cfg_noise['poisson_range'])
            
        elif ntype == 'mixed':
            # Add Gaussian then Poisson noise and record both levels.
            spec_noisy_raw, poisson_levels = self.add_PoissonNoise(spec_noisy_raw, cfg_noise['poisson_range'])
            spec_noisy_raw, gaussian_levels = self.add_GaussianNoise(spec_noisy_raw, cfg_noise['gaussian_range'])

        # Final normalization.
        spec_noisy = spec_noisy_raw
        # spec_noisy = spec_noisy_raw / np.max(spec_noisy_raw, axis=1, keepdims=True)
        bg_max = np.max(bg_total, axis=1)

        return (spec_gt, spec_noisy, gaussian_levels, poisson_levels, 
                r_num, r_pos, r_fwhm, r_int, bg_max)

# ==========================================
# Parallel worker.
# ==========================================
def worker_task(chunk_size, length, wavenumbers, cfg_phys, cfg_noise):
    # Create a generator and generate data in a worker process.
    np.random.seed() # Reset the random seed.
    gen = SpectraGenerator()
    return gen.generate_dataset(chunk_size, length, wavenumbers, cfg_phys, cfg_noise)


# ==========================================
# Main entry point.
# ==========================================
def main():
    noise_mode = CONFIG['noise']['type']
    print(f"Generating spectra with {noise_mode} noise")
    t0 = time.time()
    
    # Prepare parameters.
    total_samples = CONFIG['simulation']['num_samples']
    length = CONFIG['simulation']['length']
    wn = np.linspace(*CONFIG['simulation']['wn_range'], length)
    n_cores = min(CONFIG['system']['num_cores'], cpu_count())
    
    # Assign work batches.
    chunk_size = total_samples // n_cores
    remainder = total_samples % n_cores
    tasks = [chunk_size + 1 if i < remainder else chunk_size for i in range(n_cores)]
    
    print(f"Workers: {n_cores}, batches: {tasks}")

    # Run workers in parallel.
    worker_func = partial(worker_task, 
                          length=length, 
                          wavenumbers=wn, 
                          cfg_phys=CONFIG['physics'], 
                          cfg_noise=CONFIG['noise'])
    
    with Pool(n_cores) as pool:
        results = pool.map(worker_func, tasks)

    # Combine the generated data.
    print("Combining generated data...")
    
    # Unpack in the order returned by generate_dataset.
    (gt_list, noisy_list, g_lvls_list, p_lvls_list, 
     num_list, pos_list, fwhm_list, int_list, bg_max_list) = zip(*results)

    # Combine the spectrum arrays.
    gt = np.concatenate(gt_list, axis=0)
    noisy = np.concatenate(noisy_list, axis=0)
    r_num = np.concatenate(num_list, axis=0)
    r_pos = np.concatenate(pos_list, axis=0)
    r_fwhm = np.concatenate(fwhm_list, axis=0)
    r_int = np.concatenate(int_list, axis=0)
    bg_max = np.concatenate(bg_max_list, axis=0)

    # Combine noise levels, including absent noise types.
    # An initial None means this noise type was not generated.
    if g_lvls_list[0] is None:
        final_g_levels = np.array([]) # Store an empty array for NPZ loading.
    else:
        final_g_levels = np.concatenate(g_lvls_list, axis=0)

    if p_lvls_list[0] is None:
        final_p_levels = np.array([])
    else:
        final_p_levels = np.concatenate(p_lvls_list, axis=0)

    print(f"Elapsed: {time.time() - t0:.2f} s | Samples: {gt.shape[0]}")

    # Save files.
    save_dir = CONFIG['system']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # Build filenames from the noise and background types.
    suffix_parts = []
    
    # 1. Add the noise type.
    suffix_parts.append(noise_mode)
    
    # 2. Add the background type.
    if CONFIG['physics']['add_rayleigh']: suffix_parts.append("ray")
    if CONFIG['physics']['add_fluorescence']: suffix_parts.append("fluo")
    if not CONFIG['physics']['add_rayleigh'] and not CONFIG['physics']['add_fluorescence']:
        suffix_parts.append("clean_bg")
        
    fname_suffix = "_".join(suffix_parts) # For example: mixed_ray_fluo.

    # Save parameter statistics as NPZ.
    npz_path = os.path.join(save_dir, f'Stats_{total_samples}_{fname_suffix}.npz')
    np.savez(npz_path,
             gaussian_levels=final_g_levels,  # Save each noise level separately.
             poisson_levels=final_p_levels,   # Save each noise level separately.
             num_peaks=r_num, 
             peak_positions=r_pos, 
             peak_fwhms=r_fwhm, 
             peak_intensities=r_int, 
             bg_max=bg_max)
             
    # Save spectra as MAT.
    # mat_path = os.path.join(save_dir, f'Data_{total_samples}_{fname_suffix}.mat')
    # Use an empty array or placeholder for MATLAB compatibility.
    # mat_dict = {
    #     'GT': gt, 
    #     'Noisy': noisy, 
    #     'x': wn,
    #     'gaussian_levels': final_g_levels if final_g_levels.size > 0 else [],
    #     'poisson_levels': final_p_levels if final_p_levels.size > 0 else []
    # }
    # sio.savemat(mat_path, mat_dict)
    


if __name__ == "__main__":
    main()
