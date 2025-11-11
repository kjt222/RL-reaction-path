# -*- coding: utf-8 -*-
from pathlib import Path
from ase import io
from mace import data

def build_key_specification():
    key_spec = data.KeySpecification()
    key_spec.update(info_keys={"energy": "energy"})
    key_spec.update(arrays_keys={"forces": "force"})
    return key_spec

path = Path('d:/D/calculate/MLP/MACE项目/xyz_data/rmd17_aspirin.xyz')
frames = list(io.iread(path, index=':1000'))
train_atoms = frames[:900]
key_spec = build_key_specification()
configs = data.config_from_atoms_list(train_atoms, key_specification=key_spec)
energies = [cfg.properties.get('energy') for cfg in configs]
print('total configs', len(configs))
print('first energies', energies[:5])
missing = [i for i,e in enumerate(energies) if e is None]
print('missing count', len(missing))
print('missing indices sample', missing[:10])
