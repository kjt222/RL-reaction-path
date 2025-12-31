GemNet-OC OC22 model assets

Weights:
- gnoc_oc22_all_s2ef.pt (OC22 only)
  https://dl.fbaipublicfiles.com/opencatalystproject/models/2022_09/oc22/s2ef/gnoc_oc22_all_s2ef.pt

Config:
- gemnet_oc_oc22_local.yml (includes base_oc22_local.yml with absolute OC22 paths)
- base_oc22_local.yml
Original configs (from local FairChem tree):
- configs/oc22/s2ef/gemnet-oc/gemnet_oc.yml
- configs/oc22/s2ef/base.yml

Note:
- These configs come from the local FairChem tree and keep the original include paths.
- Use a config_dir that resolves includes (e.g., repo root or this folder as CWD).
