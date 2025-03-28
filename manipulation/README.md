
# Kortex API Installation

Download the Kortex API wheel file from the link below:

[kortex API 2.6.0.post3](https://artifactory.kinovaapps.com/artifactory/generic-public/kortex/API/2.6.0/kortex_api-2.6.0.post3-py3-none-any.whl)

Then, install it using the following command:

```bash
python3 -m pip install <whl relative fullpath name>.whl
```
e.g:

```bash
python3 -m pip install kortex_api-2.6.0.post3-py3-none-any.whl
```

Recommended python 3.10 (in a virtual environment):

```bash
#creating a virtual environment called ".venv"
python3 -m venv .venv
```

## Fix

replace:

```bash
import collections
```


with:
```bash
import collections.abc as collections
```

during the first operation, you will encounter errors like 