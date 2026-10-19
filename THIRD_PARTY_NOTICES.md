# Third-party notices

The root [LICENSE](LICENSE) applies to original Fast2Peak contributions only. It does not grant rights to third-party material or replace upstream licenses. Copyright and attribution notices in source files must be retained.

This inventory records the provenance currently available in this repository. An unresolved license is not an Apache-2.0 license, and attribution alone is not permission to redistribute. Resolve the outstanding permissions below, or exclude the affected material from the public release. This document does not certify that the entire repository is cleared for redistribution.

## IR preprocessing utilities

- Files: `otherbench/IR/utils/upstream/functions.py` and `otherbench/IR/utils/upstream/import_data.py`.
- Copyright: 2020 Josef Brandt, University of Gothenborg, as stated in the retained headers.
- License: GNU GPL version 3 or, at the recipient's option, any later version (`GPL-3.0-or-later`).
- The GPL version 3 text is included in [LICENSES/GPL-3.0.txt](LICENSES/GPL-3.0.txt). These files are excluded from the root Apache-2.0 grant. Code combined with them must satisfy the applicable GPL requirements; this notice does not override those requirements.
- Data-source attribution and permission statements are recorded separately in [the IR data description](otherbench/IR/docs/data_README.md).

## SpectraMAENet / SMAE

- Source: [pengjuRen99/SpectraMAENet](https://github.com/pengjuRen99/SpectraMAENet).
- Comparison revision: `7939f2e6d3e6e975b687e4f07e56b9ce51cae5e3`.
- Retained source and adaptations: `othermethods/smae/models/`, `othermethods/smae/utils/`, `othermethods/smae/training/` and `othermethods/smae/inference/`.
- The retained source records did not include a declared software license. No permission or license is inferred from public availability. Third-party portions and adaptations are excluded from the root Apache-2.0 grant pending permission review.
- Import/path changes and numerical/runtime adaptations are described in [the provenance record](othermethods/smae/docs/provenance.md).

## DN-Unet

- Source: [Lin Group author resources](https://lingroup.xmu.edu.cn/info/1150/1117.htm); paper [10.1021/acs.analchem.0c03087](https://doi.org/10.1021/acs.analchem.0c03087).
- Local implementation: `otherbench/NMR/models/dn_unet.py`, a PyTorch port of the author's TensorLayer architecture. Associated author-derived weights are stored separately.
- The retained records do not establish a software/weight redistribution license. The port and author-derived weights are excluded from the root Apache-2.0 grant pending review.
- Architecture conversion and fine-tuning details are in [the NMR provenance record](otherbench/NMR/docs/model_provenance.md).

## LD-Net

- Source: [HaolinZhan/LD-Net-for-NMR-spectroscopy-denoising](https://github.com/HaolinZhan/LD-Net-for-NMR-spectroscopy-denoising), revision `38130d612b8cb02e61c48afbad5cf6c30db9fabf`.
- Paper: [10.3866/PKU.WHXB202310045](https://doi.org/10.3866/PKU.WHXB202310045).
- Local implementation: `otherbench/NMR/models/ld_net.py`; the retained model and block classes omit unused plotting imports. Author-derived weights are stored separately.
- The retained records do not establish a software/weight redistribution license. These third-party portions and author-derived weights are excluded from the root Apache-2.0 grant pending review. See [the NMR provenance record](otherbench/NMR/docs/model_provenance.md).

## Other baseline implementations

The source provenance and permission scope of `othermethods/peak2peak/`, `models/ViT_1D.py`, `models/aunet.py`, `models/resunet.py`, `models/raman_mae.py`, `models/smae.py` and `models/smae2d.py` still need an explicit file-level review. These baseline files are excluded from the root Apache-2.0 grant until their original and third-party portions are identified and the applicable terms are documented. This exclusion is not a claim that all of these files were copied from upstream sources.

## Fonts

`utils/Arial.ttf` is a third-party font, not Fast2Peak source code, and is excluded from the root Apache-2.0 grant. Its redistribution permission has not been established in the retained records. Verify that permission or omit the font from the public release and use a user-installed or appropriately licensed alternative.

## Datasets and model weights

The root Apache-2.0 grant does not cover datasets or model weights. Their release must include separate terms consistent with the original sources and any applicable permissions. See the [IR](otherbench/IR/docs/data_README.md), [NMR](otherbench/NMR/docs/data_README.md) and [XRD](otherbench/XRD/docs/data_README.md) provenance records. Planned Hugging Face uploads must carry their own dataset/model licensing information; no license is inferred from their intended hosting location.

## Installed dependencies

Dependencies listed in `requirements.txt` retain their respective licenses. Installing them does not relicense them under Apache-2.0. If dependency code or binaries are later redistributed with Fast2Peak, include the notices required by their licenses.
