![Logo](assets/logo.svg)
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm-project.org)
[![PyPI](https://img.shields.io/pypi/v/pdm_template_uq?logo=python&logoColor=%23cccccc)](https://pypi.org/project/pdm_template_uq)
[![Python 3.11](https://img.shields.io/badge/python-3.11+-blue.svg?logo=python&logoColor=cccccc)](https://www.python.org/downloads/)

## Installation
You can install normally with:
```shell
pip install picrom
```
If you are using [pdm](https://github.com/pdm-project/pdm) in your own project, then you can use:
```shell
cd <your-pdm-project>
pdm add picrom
```
You can also quickly set up a development environment with:
```shell
# After forking this project on Github...
git clone https://github.com/<your-username>/picrom.git
cd picrom
pdm install  # reads pdm.lock and sets up an identical venv
```

## Quickstart
```python
import picrom

picrom.do_something()

print('Wow!')
```

## Contributing
See the [contribution](CONTRIBUTING.md) guidelines.

## Citations
Include any additional references for your project.

<sup><sub>Made with the [UQ pdm template](https://github.com/eckelsjd/picrom.git).</sub></sup>

