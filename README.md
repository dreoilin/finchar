<h1 align="center">finchar</h1>

## Table of contents

- [About](#about)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [Authors](#authors)

## About

finchar is a Python 3 gm/ID starter kit for Finfet devices.

## Installation

To install finchar from source, download from Github and run pip:

`pip install .`

 in the root directory.

## Usage

### Scripting with the Lookup Class
A gm/ID lookup object can be generated with the `Lookup` class. The lookup object requires lookup data for initialisation. Both `.mat` files generated using MATLAB or `.pkl` files generated using finchar's own characterisation script are supported.

You can create a lookup object as follows:

```python
from finchar import Lookup as lk

NCH = lk('NCH_char.pkl')
```
### Access MOS Data
The `Lookup` class allows for pseudo array access of the MOS matrix data. You can access data as follows:

```python
# get VGS data as array from NCH
VGS_array = NCH['VGS']
```

Data is returned as a deep copy of the array contained in the `Lookup` object.

### Lookup functionality 

Lookup of interpolated data occurs as follows:

```python
VDSs = NCH['VDS'] 
VGSs = np.arange(0.4, 0.6, 0.05)
# Plot ID versus VDS
ID = NCH.look_up('ID', vds=VDSs, vgs=VGSs)
# alias function lookup can also be used
ID = NCH.lookup('ID', vds=VDSs, vgs=VGSs)
# check bias
VGS = NCH.look_upVGS(GM_ID = 10, VDS = 0.6, VSB = 0.1, L = 0.18)
print(f'VGS is: {VGS}')

plt.plot(VDSs, np.transpose(ID))
```

Modes 1 (Simple parameter lookup), mode 2 (arbitrary ratio lookup) and mode 3 (cross lookup of ratios) are implemented. The companion lookupVGS function is also included.

### Sweeping a Technology

`finchar` features a CLI which can be used to run techweeps to generate transistor data.

## Authors

- Cian O'Donnell : cian.odonnell@mcci.ie
