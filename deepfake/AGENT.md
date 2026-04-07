# Agent instructions

Conventions for this codebase. Follow them when adding or editing code.

## Imports

Import **modules** at the top of the file. Do not import **classes** (or other symbols) from a package only to instantiate them in one place—import the module and qualify the name.

**Prefer:**

```python
import dataset as dataset_module

dataset = dataset_module.Dataset()
```

**Avoid:**

```python
from dataset import Dataset

dataset = Dataset()
```

Rationale: module-level imports keep namespaces explicit and make it obvious where types come from when reading the file.

## Variable names

Use **full, descriptive names** instead of abbreviations.

| Prefer   | Avoid |
|----------|-------|
| `config` | `cfg` |
| `encoder` | `enc` |
| `decoder` | `dec` |

Short single-letter names are acceptable for **local mathematical or tensor convention**, for example `x`, `y`, `z`, `i`, `j`, loop indices, or coordinates.

```python
for i, batch in enumerate(loader):
    x, y = batch  # fine
```

When in doubt, choose clarity over brevity.

## Configuration
Use omegaconf to load configuration yaml files. 

If a configuration variable is needed in two locations us `$var` in the yaml.
ie
```yaml
encoder:
  channels: 32

decoder:
  channels: $encoder.size
```