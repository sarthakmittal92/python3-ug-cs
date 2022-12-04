#!/bin/bash

python3 -m pip install --upgrade twine 1> /dev/null
python3 -m twine upload --verbose --repository testpypi dist/* -u __token__