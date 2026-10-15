"""Evaluate the exact Table S4 method set on the native-6400 Maybridge benchmark."""

import argparse
import gc
import json
import time
from pathlib import Path
import h5py
import numpy as np
import torch

from otherbench.NMR.utils.config import DEFAULT_CONFIG, NMR_ROOT, load_config
from otherbench.NMR.utils.metrics import metrics_for_sample
from otherbench.NMR.utils.reporting import save_results
from otherbench.NMR.models.baselines import wavelet_denoise, cadzow_pseudo_fid
from otherbench.NMR.models.dn_unet import load_dn_unet
from otherbench.NMR.models.ld_net import load_ld_net
from otherbench.NMR.models.f2p import load_f2p
from otherbench.NMR.inference.networks import infer_author_network
from otherbench.NMR.inference.f2p import infer_f2p


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, default=NMR_ROOT / "datasets/benchmark/benchmark.h5")
    parser.add_argument("--output-dir", type=Path, default=NMR_ROOT / "outputs/evaluation")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--skip-f2p", action="store_true")
    parser.add_argument("--skip-author-networks", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    f2p, checkpoint = load_f2p(config, device) if not args.skip_f2p else (None, None)
    dn, ld = None, None
    if not args.skip_author_networks:
        dn_config = config["dn_unet"]
        dn = load_dn_unet(Path(), Path(dn_config["converted_checkpoint"]), device,
                          Path(dn_config["trained_checkpoint"]))
        gc.collect()
        ld = load_ld_net(config, device)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, times = [], {}
    with h5py.File(args.data) as source, h5py.File(output / "denoised.h5", "w") as dest:
        count = len(source["gt"]) if args.max_samples is None else min(args.max_samples, len(source["gt"]))
        if count < 1:
            raise ValueError("At least one sample is required")
        target = source["gt"][:count].astype(np.float32)
        ppm = source["ppm"][:].astype(np.float32)
        dest.create_dataset("gt", data=target, compression="gzip", shuffle=True)
        dest.create_dataset("ppm", data=ppm)
        groups = dest.create_group("conditions")
        for condition in config["noise_conditions"]:
            name = condition["name"]
            noisy = source["noise"][name][:count].astype(np.float32)
            started = time.time()
            methods = {"RAW": noisy,
                       "WT": wavelet_denoise(noisy, config["baselines"]["wavelet"]),
                       "Cadzow": cadzow_pseudo_fid(noisy, config["baselines"]["cadzow"])}
            times[name] = {"classical_seconds": time.time() - started}
            for method, model, key in (("LD-Net", ld, "ld_net"), ("DN-Unet", dn, "dn_unet")):
                if model is not None:
                    options = config[key]
                    methods[method], times[name][method] = infer_author_network(
                        model, noisy, int(options["native_length"]), int(options["native_length"]),
                        int(options["batch_size"]), device, options["normalization"])
            if f2p is not None:
                methods["F2P"], times[name]["F2P"] = infer_f2p(
                    f2p, noisy, 6400, 6400, int(config["f2p"]["batch_size"]), device)
            group = groups.create_group(name)
            group.create_dataset("noisy", data=noisy, compression="gzip", shuffle=True)
            for method, prediction in methods.items():
                if method != "RAW":
                    group.create_dataset(method, data=prediction, compression="gzip", shuffle=True)
                for index in range(count):
                    rows.append({"condition": name, "method": method, "sample_index": index,
                                 **metrics_for_sample(target[index], prediction[index], noisy[index],
                                                      ppm, config["peak_metrics"])})
            print(f"{name}: {count} spectra, {len(methods)} methods", flush=True)
    leaderboard, overall = save_results(rows, output)
    summary = {"samples_per_condition": count, "noise_conditions": config["noise_conditions"],
               "device": str(device), "f2p_checkpoint_epoch": checkpoint.get("epoch") if checkpoint else None,
               "method_configuration": {k: config[k] for k in ("f2p", "dn_unet", "ld_net", "baselines", "peak_metrics")},
               "inference_times": times, "leaderboard": leaderboard, "overall": overall}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
