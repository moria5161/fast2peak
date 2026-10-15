# ATR reference spectra

`../datasets/atr_spectra.npy` stores all 174 reference spectra as columns in a numeric NumPy array.

- Shape: `(1863, 175)`.
- Data type: `float64`; load with `allow_pickle=False`.
- NumPy column `0` (the first physical column): wavenumber in cm⁻¹.
- NumPy columns `1–174` (the second through 175th physical columns): spectral intensity.
- Rows follow increasing wavenumber, from 401.2 to 3992.6 cm⁻¹.
- Spectrum columns are sorted by the source filename, case-insensitively; exact filenames break ties.
- Material names are the original filename stems. Original codes and spelling are retained.
- Intensities are not normalized and no noise is added during export.

## Alignment of different source axes

The common grid is copied from `1_2_polybutadiene.csv`, with 1,863 points. The 165 spectra sharing this exact grid are copied directly. The nine other spectra (`PA6.CSV`, `PE.csv`, `PET.csv`, `PHB.CSV`, `PLLA.CSV`, `PMMA.CSV`, `PP.csv`, `PS.csv`, `PVC.csv`) are mapped to this grid using the nearest original wavenumber, matching the existing `importData.load_specCSVs_from_directory` rule. The original CSVs have been archived outside the repository; their historical filenames remain in the table for provenance. The common-grid NPY representation is stored in `../datasets/atr_spectra.npy` (Git-ignored).

For `PE.csv` (NumPy column 83) and `PVC.csv` (NumPy column 154), 142 common-grid points below the original measured range are assigned the closest endpoint intensity. These are endpoint-filled values, not measurements at those wavenumbers. The README marks both columns below; consult the archived original CSVs when the low-wavenumber coverage matters.

The frozen 110-reference selection, paired benchmark data and published results are unchanged by this export.

## Loading

```python
import numpy as np

data = np.load("otherbench/IR/datasets/atr_spectra.npy", allow_pickle=False)
wavenumbers = data[:, 0]   # (1863,)
spectra = data[:, 1:]     # (1863, 174), one spectrum per column
first_spectrum = data[:, 1]  # 1_2_polybutadiene
```

## Column-to-material mapping

All indices below are **zero-based NumPy column indices**. Column 0 is the wavenumber axis; each following column corresponds to one source spectrum.

| NumPy column | Material / spectrum label | Original CSV (provenance) | Alignment |
|---:|---|---|---|
| 0 | Wavenumber (cm⁻¹) | `1_2_polybutadiene.csv` | Common grid |
| 1 | `1_2_polybutadiene` | `1_2_polybutadiene.csv` | Direct copy |
| 2 | `acrylonitrile_butadiene` | `acrylonitrile_butadiene.csv` | Direct copy |
| 3 | `acrylonitrile_butadiene_styrene` | `acrylonitrile_butadiene_styrene.csv` | Direct copy |
| 4 | `algae laminaria sacharina` | `algae laminaria sacharina.csv` | Direct copy |
| 5 | `algae_desmarestia_viridis` | `algae_desmarestia_viridis.csv` | Direct copy |
| 6 | `algae_fucus_serratus` | `algae_fucus_serratus.csv` | Direct copy |
| 7 | `algae_laminaria_digita_and_hyperborea` | `algae_laminaria_digita_and_hyperborea.csv` | Direct copy |
| 8 | `alginic_acid_ sodium_salt` | `alginic_acid_ sodium_salt.csv` | Direct copy |
| 9 | `alkyd_varnish` | `alkyd_varnish.csv` | Direct copy |
| 10 | `amber` | `amber.csv` | Direct copy |
| 11 | `aramid` | `aramid.csv` | Direct copy |
| 12 | `black_broodcomb` | `black_broodcomb.csv` | Direct copy |
| 13 | `broodcomb` | `broodcomb.csv` | Direct copy |
| 14 | `broodcomb_once_brooded` | `broodcomb_once_brooded.csv` | Direct copy |
| 15 | `butyl_methacrylate_isobutyl_methacrolate` | `butyl_methacrylate_isobutyl_methacrolate.csv` | Direct copy |
| 16 | `cellulose` | `cellulose.csv` | Direct copy |
| 17 | `cellulose_acetate` | `cellulose_acetate.csv` | Direct copy |
| 18 | `cellulose_acetate_butyrate` | `cellulose_acetate_butyrate.csv` | Direct copy |
| 19 | `cellulose_propionate` | `cellulose_propionate.csv` | Direct copy |
| 20 | `cellulose_wipe` | `cellulose_wipe.csv` | Direct copy |
| 21 | `chitin_cancer_pagurus` | `chitin_cancer_pagurus.csv` | Direct copy |
| 22 | `chitin_crangon_antonia` | `chitin_crangon_antonia.csv` | Direct copy |
| 23 | `chitin_from_crustacean_shells` | `chitin_from_crustacean_shells.csv` | Direct copy |
| 24 | `cigarette_filter` | `cigarette_filter.csv` | Direct copy |
| 25 | `coal` | `coal.csv` | Direct copy |
| 26 | `copolyamide` | `copolyamide.csv` | Direct copy |
| 27 | `copolyester` | `copolyester.csv` | Direct copy |
| 28 | `crangon _chitin_exuvie` | `crangon _chitin_exuvie.csv` | Direct copy |
| 29 | `epoxide_resin` | `epoxide_resin.csv` | Direct copy |
| 30 | `ethyl_cellulose` | `ethyl_cellulose.csv` | Direct copy |
| 31 | `ethylene_acrylic_acid` | `ethylene_acrylic_acid.csv` | Direct copy |
| 32 | `ethylene_ethyl_acrylate` | `ethylene_ethyl_acrylate.csv` | Direct copy |
| 33 | `ethylene_propylene` | `ethylene_propylene.csv` | Direct copy |
| 34 | `ethylene_vinyl_acetate` | `ethylene_vinyl_acetate.csv` | Direct copy |
| 35 | `ethylene_vinyl_alcohol` | `ethylene_vinyl_alcohol.csv` | Direct copy |
| 36 | `fibre_acetate` | `fibre_acetate.csv` | Direct copy |
| 37 | `fibre_cocoanut` | `fibre_cocoanut.csv` | Direct copy |
| 38 | `fibre_cotton_combers` | `fibre_cotton_combers.csv` | Direct copy |
| 39 | `fibre_cotton_US_pima` | `fibre_cotton_US_pima.csv` | Direct copy |
| 40 | `fibre_cotton_uzbekistan` | `fibre_cotton_uzbekistan.csv` | Direct copy |
| 41 | `fibre_flax` | `fibre_flax.csv` | Direct copy |
| 42 | `fibre_grass` | `fibre_grass.csv` | Direct copy |
| 43 | `fibre_hemp_fine` | `fibre_hemp_fine.csv` | Direct copy |
| 44 | `fibre_indian_raw_cotton` | `fibre_indian_raw_cotton.csv` | Direct copy |
| 45 | `fibre_kapok` | `fibre_kapok.csv` | Direct copy |
| 46 | `fibre_mulberry_silk` | `fibre_mulberry_silk.csv` | Direct copy |
| 47 | `fibre_polyactide` | `fibre_polyactide.csv` | Direct copy |
| 48 | `fibre_polyamide_6` | `fibre_polyamide_6.csv` | Direct copy |
| 49 | `fibre_polyamide_6_P6.6` | `fibre_polyamide_6_P6.6.csv` | Direct copy |
| 50 | `fibre_polyester` | `fibre_polyester.csv` | Direct copy |
| 51 | `fibre_polyetheretherketone` | `fibre_polyetheretherketone.csv` | Direct copy |
| 52 | `fibre_polypropylene` | `fibre_polypropylene.csv` | Direct copy |
| 53 | `fibre_polypropylene_dyed` | `fibre_polypropylene_dyed.csv` | Direct copy |
| 54 | `fibre_polyvinyl_alcohol` | `fibre_polyvinyl_alcohol.csv` | Direct copy |
| 55 | `fibre_polyvinylidene_fluoride` | `fibre_polyvinylidene_fluoride.csv` | Direct copy |
| 56 | `fibre_poplar_down` | `fibre_poplar_down.csv` | Direct copy |
| 57 | `fibre_roasted_flax` | `fibre_roasted_flax.csv` | Direct copy |
| 58 | `fibre_silk_slubbing` | `fibre_silk_slubbing.csv` | Direct copy |
| 59 | `fibre_thermoplastic_elastomere` | `fibre_thermoplastic_elastomere.csv` | Direct copy |
| 60 | `fibre_turf` | `fibre_turf.csv` | Direct copy |
| 61 | `fibre_tussah_silk` | `fibre_tussah_silk.csv` | Direct copy |
| 62 | `fibre_urtica_dioica_L_conar_fibra` | `fibre_urtica_dioica_L_conar_fibra.csv` | Direct copy |
| 63 | `fibre_viscose` | `fibre_viscose.csv` | Direct copy |
| 64 | `fibre_viscose_dyed` | `fibre_viscose_dyed.csv` | Direct copy |
| 65 | `fur_alpaca` | `fur_alpaca.csv` | Direct copy |
| 66 | `fur_yak_bleached` | `fur_yak_bleached.csv` | Direct copy |
| 67 | `honeycomb` | `honeycomb.csv` | Direct copy |
| 68 | `hydroxyethyl_cellulose` | `hydroxyethyl_cellulose.csv` | Direct copy |
| 69 | `hydroxypropyl_cellulose` | `hydroxypropyl_cellulose.csv` | Direct copy |
| 70 | `hydroxypropyl_methyl_cellulose` | `hydroxypropyl_methyl_cellulose.csv` | Direct copy |
| 71 | `lahmian_medium_acrylic_paint` | `lahmian_medium_acrylic_paint.csv` | Direct copy |
| 72 | `merino_scoured_wool_made_rough` | `merino_scoured_wool_made_rough.csv` | Direct copy |
| 73 | `methyl_vinyl_ether_maleic_acid` | `methyl_vinyl_ether_maleic_acid.csv` | Direct copy |
| 74 | `methyl_vinyl_ether_maleic_anhydride` | `methyl_vinyl_ether_maleic_anhydride.csv` | Direct copy |
| 75 | `n_vinylpyrrolidone_vinyl_acetate` | `n_vinylpyrrolidone_vinyl_acetate.csv` | Direct copy |
| 76 | `nitrile_rubber` | `nitrile_rubber.csv` | Direct copy |
| 77 | `nylon_11` | `nylon_11.csv` | Direct copy |
| 78 | `nylon_12` | `nylon_12.csv` | Direct copy |
| 79 | `nylon_6` | `nylon_6.csv` | Direct copy |
| 80 | `nylon_6_12` | `nylon_6_12.csv` | Direct copy |
| 81 | `nylon_6_6` | `nylon_6_6.csv` | Direct copy |
| 82 | `PA6` | `PA6.CSV` | Nearest neighbor |
| 83 | `PE` | `PE.csv` | Nearest neighbor; 142 endpoint-filled points |
| 84 | `PET` | `PET.csv` | Nearest neighbor |
| 85 | `PHB` | `PHB.CSV` | Nearest neighbor |
| 86 | `phenoxy_resin` | `phenoxy_resin.csv` | Direct copy |
| 87 | `PLLA` | `PLLA.CSV` | Nearest neighbor |
| 88 | `PMMA` | `PMMA.CSV` | Nearest neighbor |
| 89 | `poly(1_butene)_isotactic` | `poly(1_butene)_isotactic.csv` | Direct copy |
| 90 | `poly(2_4_6_tribromostyrene)` | `poly(2_4_6_tribromostyrene).csv` | Direct copy |
| 91 | `poly(2_6_dimethyl_p_phenylene_oxide)` | `poly(2_6_dimethyl_p_phenylene_oxide).csv` | Direct copy |
| 92 | `poly(2_hydroxyethyl_methacrylate)` | `poly(2_hydroxyethyl_methacrylate).csv` | Direct copy |
| 93 | `poly(4_4'_dipropoxy_2_2'_diphenyl_propane_fumarate)` | `poly(4_4'_dipropoxy_2_2'_diphenyl_propane_fumarate).csv` | Direct copy |
| 94 | `poly(4_methyl_1_pentene)` | `poly(4_methyl_1_pentene).csv` | Direct copy |
| 95 | `poly(acrylic_acid)` | `poly(acrylic_acid).csv` | Direct copy |
| 96 | `poly(butylene_terephthalate)` | `poly(butylene_terephthalate).csv` | Direct copy |
| 97 | `poly(diallyl_isophthalate)` | `poly(diallyl_isophthalate).csv` | Direct copy |
| 98 | `poly(ethyl_methacrylate)` | `poly(ethyl_methacrylate).csv` | Direct copy |
| 99 | `poly(ethylene_glycol)` | `poly(ethylene_glycol).csv` | Direct copy |
| 100 | `poly(ethylene_oxide)` | `poly(ethylene_oxide).csv` | Direct copy |
| 101 | `poly(isobutyl_methacrylate)` | `poly(isobutyl_methacrylate).csv` | Direct copy |
| 102 | `poly(p_phenylene_ether_sulphone)` | `poly(p_phenylene_ether_sulphone).csv` | Direct copy |
| 103 | `poly(phenylene_sulfide)` | `poly(phenylene_sulfide).csv` | Direct copy |
| 104 | `poly(tetrafluoroethylene)` | `poly(tetrafluoroethylene).csv` | Direct copy |
| 105 | `poly(vinyl_acetate)` | `poly(vinyl_acetate).csv` | Direct copy |
| 106 | `poly(vinyl_alcohol)` | `poly(vinyl_alcohol).csv` | Direct copy |
| 107 | `poly(vinyl_butyral)` | `poly(vinyl_butyral).csv` | Direct copy |
| 108 | `poly(vinyl_chloride)_carboxylated` | `poly(vinyl_chloride)_carboxylated.csv` | Direct copy |
| 109 | `poly(vinyl_formal)` | `poly(vinyl_formal).csv` | Direct copy |
| 110 | `poly(vinyl_stearate)` | `poly(vinyl_stearate).csv` | Direct copy |
| 111 | `poly(vinylidene_fluoride)` | `poly(vinylidene_fluoride).csv` | Direct copy |
| 112 | `polyacetal` | `polyacetal.csv` | Direct copy |
| 113 | `polyacrylamide` | `polyacrylamide.csv` | Direct copy |
| 114 | `polyacrylamide_carboxyl_modified` | `polyacrylamide_carboxyl_modified.csv` | Direct copy |
| 115 | `polyamide` | `polyamide.csv` | Direct copy |
| 116 | `polyamide_6` | `polyamide_6.csv` | Direct copy |
| 117 | `polyamide_resin` | `polyamide_resin.csv` | Direct copy |
| 118 | `polycaprolactone` | `polycaprolactone.csv` | Direct copy |
| 119 | `polycarbonate` | `polycarbonate.csv` | Direct copy |
| 120 | `polychloroprene` | `polychloroprene.csv` | Direct copy |
| 121 | `polyester` | `polyester.csv` | Direct copy |
| 122 | `polyester_epoxide` | `polyester_epoxide.csv` | Direct copy |
| 123 | `polyesterurethane` | `polyesterurethane.csv` | Direct copy |
| 124 | `polyetherester` | `polyetherester.csv` | Direct copy |
| 125 | `polyethylene_chlorinated` | `polyethylene_chlorinated.csv` | Direct copy |
| 126 | `polyethylene_chlorosulfonated` | `polyethylene_chlorosulfonated.csv` | Direct copy |
| 127 | `polyethylene_foamed` | `polyethylene_foamed.csv` | Direct copy |
| 128 | `polyethylene_high_density` | `polyethylene_high_density.csv` | Direct copy |
| 129 | `polyethylene_low_density` | `polyethylene_low_density.csv` | Direct copy |
| 130 | `polyethylene_low_density_linear` | `polyethylene_low_density_linear.csv` | Direct copy |
| 131 | `polyethylene_wax` | `polyethylene_wax.csv` | Direct copy |
| 132 | `polyethylene_wax_oxidized` | `polyethylene_wax_oxidized.csv` | Direct copy |
| 133 | `polyhydroxybutyric_acid` | `polyhydroxybutyric_acid.csv` | Direct copy |
| 134 | `polyimide` | `polyimide.csv` | Direct copy |
| 135 | `polyisoprene_chlorinated` | `polyisoprene_chlorinated.csv` | Direct copy |
| 136 | `polylactic_acid` | `polylactic_acid.csv` | Direct copy |
| 137 | `polymaide_66` | `polymaide_66.csv` | Direct copy |
| 138 | `polymethyl methacrylate` | `polymethyl methacrylate.csv` | Direct copy |
| 139 | `polyoxymethylene` | `polyoxymethylene.csv` | Direct copy |
| 140 | `polyphenylsulfone` | `polyphenylsulfone.csv` | Direct copy |
| 141 | `polypropylene` | `polypropylene.csv` | Direct copy |
| 142 | `polypropylene_isotactic` | `polypropylene_isotactic.csv` | Direct copy |
| 143 | `polystyrene` | `polystyrene.csv` | Direct copy |
| 144 | `polystyrene_expanded` | `polystyrene_expanded.csv` | Direct copy |
| 145 | `polysulfone` | `polysulfone.csv` | Direct copy |
| 146 | `polytehylene_terephthalate` | `polytehylene_terephthalate.csv` | Direct copy |
| 147 | `polyurethane` | `polyurethane.csv` | Direct copy |
| 148 | `polyurethane_acrylic_resin` | `polyurethane_acrylic_resin.csv` | Direct copy |
| 149 | `polyvinylchloride` | `polyvinylchloride.csv` | Direct copy |
| 150 | `polyvinylchloride_with_plasticizer` | `polyvinylchloride_with_plasticizer.csv` | Direct copy |
| 151 | `polyvinylpyrrolidone` | `polyvinylpyrrolidone.csv` | Direct copy |
| 152 | `PP` | `PP.csv` | Nearest neighbor |
| 153 | `PS` | `PS.csv` | Nearest neighbor |
| 154 | `PVC` | `PVC.csv` | Nearest neighbor; 142 endpoint-filled points |
| 155 | `resin_dispersion` | `resin_dispersion.csv` | Direct copy |
| 156 | `scoured wool_ not_made_rough` | `scoured wool_ not_made_rough.csv` | Direct copy |
| 157 | `sealing_ring_EPDM` | `sealing_ring_EPDM.csv` | Direct copy |
| 158 | `sealing_ring_Gardena_1124_large` | `sealing_ring_Gardena_1124_large.csv` | Direct copy |
| 159 | `silicone_rubber` | `silicone_rubber.csv` | Direct copy |
| 160 | `silicone_seal_reactor` | `silicone_seal_reactor.csv` | Direct copy |
| 161 | `styrene_acrylonitrile` | `styrene_acrylonitrile.csv` | Direct copy |
| 162 | `styrene_allyl_alcohol` | `styrene_allyl_alcohol.csv` | Direct copy |
| 163 | `styrene_butadiene` | `styrene_butadiene.csv` | Direct copy |
| 164 | `styrene_butyl_methacrylate` | `styrene_butyl_methacrylate.csv` | Direct copy |
| 165 | `styrene_ethylene_butylene` | `styrene_ethylene_butylene.csv` | Direct copy |
| 166 | `styrene_isoprene` | `styrene_isoprene.csv` | Direct copy |
| 167 | `styrene_maleic_anhydride` | `styrene_maleic_anhydride.csv` | Direct copy |
| 168 | `vinyl_chloride_vinyl_acetate` | `vinyl_chloride_vinyl_acetate.csv` | Direct copy |
| 169 | `vinyl_chloride_vinyl_acetate_hydroxypropyl_acrylate` | `vinyl_chloride_vinyl_acetate_hydroxypropyl_acrylate.csv` | Direct copy |
| 170 | `vinyl_chloride_vinyl_acetate_maleic_acid` | `vinyl_chloride_vinyl_acetate_maleic_acid.csv` | Direct copy |
| 171 | `vinylidene_chloride_acrylonitrile` | `vinylidene_chloride_acrylonitrile.csv` | Direct copy |
| 172 | `vinylidene_chloride_vinyl_chloride` | `vinylidene_chloride_vinyl_chloride.csv` | Direct copy |
| 173 | `windscreen_wiper_rubber` | `windscreen_wiper_rubber.csv` | Direct copy |
| 174 | `wool` | `wool.csv` | Direct copy |

## Regeneration and provenance

The NPY is ready to load. Re-exporting it requires the original CSVs, which are archived separately. From the repository root, with the `fast2peak` environment active:

```bash
python otherbench/IR/main.py pack \
    --input-dir path/to/original_csvs \
    --output "otherbench/IR/datasets/atr_spectra.npy"
```

The script prints the complete column mapping as JSON and writes the NPY. If source CSV filenames are added or removed, update this column table to match the regenerated mapping.

## Data source

The ATR Reference Spectra were obtained with permission from [simple-plastics.eu/download.html](https://simple-plastics.eu/download.html).

The reference databases were first described in:

Primpke, S.; Wirth, M.; Lorenz, C.; Gerdts, G. "Reference database design for the automated analysis of microplastic samples based on Fourier transform infrared (FTIR) spectroscopy." *Anal. Bioanal. Chem.* **2018**, *410*, 5131–5141.

This section incorporates the original `ATR Reference Spectra Source.txt` statement. It does not assign a new redistribution license to the spectra.
