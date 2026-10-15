import os
import h5py
import numpy as np
import argparse

# ==========================================
# Extract spectra and combine files.
# ==========================================

def process_and_combine(input_raw_dir, temp_dir, h5_output_path):
    # Extract spectra from raw TXT files and combine them into H5.
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # 1. Extract spectra into temporary files.
    print(f"Extracting spectra from: {input_raw_dir}")
    raw_files = sorted([f for f in os.listdir(input_raw_dir) if f.endswith('.txt')])
    
    for filename in raw_files:
        input_path = os.path.join(input_raw_dir, filename)
        output_path = os.path.join(temp_dir, filename)
        
        processed_data = []
        start_reading = False
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if 'Raman Shift' in line and 'Raw data' in line:
                    start_reading = True
                    continue
                if start_reading:
                    parts = line.strip().split(';')
                    if len(parts) >= 7:
                        try:
                            # Column 3: Raman Shift; column 6: Raw data #1.
                            processed_data.append([float(parts[3]), float(parts[6])])
                        except ValueError: continue
        
        if processed_data:
            np.savetxt(output_path, np.array(processed_data), fmt='%.6f', delimiter='\t')

    # 2. Combine the extracted TXT files into H5.
    print(f"Writing H5 file: {h5_output_path}")
    processed_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.txt')])
    
    all_intensities = []
    waves = None

    for filename in processed_files:
        data = np.loadtxt(os.path.join(temp_dir, filename))
        if waves is None:
            waves = data[:, 0]
        all_intensities.append(data[:, 1])

    noisy_data = np.array(all_intensities)

    with h5py.File(h5_output_path, 'w') as hf:
        hf.create_dataset('waves', data=waves, chunks=True, compression="gzip")
        hf.create_dataset('noisy', data=noisy_data, chunks=True, compression="gzip")
    
    print(f"Saved spectra: {noisy_data.shape}")


# ==========================================
# Decode H5 spectra into TXT files.
# ==========================================

def decode_h5_to_txt(h5_path, output_dir, data_key='denoised'):
    # Export each spectrum as TXT, using sample indices if waves is absent.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Reading {data_key} from: {h5_path}")
    
    with h5py.File(h5_path, 'r') as hf:
        # Check that the requested data key exists.
        if data_key not in hf:
            print(f"Error: key '{data_key}' not found in {h5_path}")
            return
        
        data_matrix = hf[data_key][:]  # Shape: batch * length.
        length = data_matrix.shape[1]  # Spectrum length.
        
        # Use the stored wavenumber axis when available.
        if 'waves' in hf:
            waves = hf['waves'][:]
        else:
            # Otherwise generate sample indices [0.0, 1.0, 2.0, ...].
            waves = np.arange(length, dtype=float)

        for i in range(data_matrix.shape[0]):
            # Generate filenames such as sample_0000.txt.
            out_filename = os.path.join(output_dir, f"sample_{i:04d}.txt")
            
            # Combine the axis and intensity for this spectrum.
            # column_stack combines two 1D arrays into (N, 2).
            combined = np.column_stack((waves, data_matrix[i]))
            
            np.savetxt(out_filename, combined, fmt='%.6f', delimiter='\t')
            
    print(f"Saved {data_matrix.shape[0]} TXT files to: {output_dir}")


# ==========================================
# Main entry point.
# ==========================================

if __name__ == "__main__":
    # Paths.
    RAW_DIR = '/home/slurmusers/moria/codes/fast2peak_v2/datasets/anjisuan'
    TEMP_DIR = '/home/slurmusers/moria/codes/fast2peak_v2/datasets/anjisuan2'
    H5_FILE = '/home/slurmusers/moria/codes/fast2peak_v2/smae/experiments/smae with duotai ss2048 ps16/duotai.h5'
    DECODE_DIR = '/home/slurmusers/moria/codes/fast2peak_v2/datasets/smae_duotai_results'

    # 1. Convert raw TXT files to H5.
    # process_and_combine(RAW_DIR, TEMP_DIR, H5_FILE)

    # 2. Example: decode H5 into TXT files.
    # Decode the 'noisy' key; use 'denoised' for predictions.
    decode_h5_to_txt(H5_FILE, DECODE_DIR, data_key='denoised')
