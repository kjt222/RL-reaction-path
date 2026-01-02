# MACE-MH-1 OC22 head

- Base model.json: `models/MACE-MH-1/manifest_bundle/model.json`
- Base weights: `models/MACE-MH-1/raw/mace-mh-1.model`
- New head: `oc22`
- Fallback head for missing elements: `omat_pbe`
- E0 fit LMDB: `Data/oc22_data/oc22_data/s2ef-total/train` (max_samples=8225293, FP64)
- z_table length: 89 (only elements present in OC22 were updated)

Usage (core trainer):
- `train.input_json`: `models/MACE-MH-1/OC22/model.json`
- `model_in`: `models/MACE-MH-1/raw/mace-mh-1.model`
- `train.head_key`: `oc22`
- `train.freeze`: `head_only`
