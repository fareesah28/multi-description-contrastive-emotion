#!/bin/bash

# load Hyperion environment
source /opt/flight/etc/setup.sh
flight env activate gridware
module add compilers/gcc gnu

# setup pyenv and virtualenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"

# only clone pyenv if not already installed
if [ ! -d "$PYENV_ROOT" ]; then
    git clone https://github.com/pyenv/pyenv.git "$PYENV_ROOT"
    git clone https://github.com/pyenv/pyenv-virtualenv.git "$PYENV_ROOT/plugins/pyenv-virtualenv"
fi

# initialise pyenv
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

if ! pyenv versions | grep -q "3.9.5"; then
    https_proxy=http://hpc-proxy00.city.ac.uk:3128 \
    CPPFLAGS="-I/opt/apps/gnu/include" \
    LDFLAGS="-L/opt/apps/gnu/lib -L/opt/apps/gnu/lib64 -ltinfo" \
    pyenv install 3.9.5
fi

# create thesis_env if it doesn't exist
if ! pyenv virtualenvs | grep -q "thesis_env"; then
    pyenv virtualenv 3.9.5 thesis_env
fi

# activate thesis_env 
pyenv activate thesis_env
echo "thesis_env" > "$HOME/.python-version"

export http_proxy=http://hpc-proxy00.city.ac.uk:3128
export https_proxy=http://hpc-proxy00.city.ac.uk:3128
export HTTP_PROXY=http://hpc-proxy00.city.ac.uk:3128
export HTTPS_PROXY=http://hpc-proxy00.city.ac.uk:3128

export PIP_CACHE_DIR=/tmp/pip_cache
export HF_HOME=/tmp/hf_home
export TRANSFORMERS_CACHE=/tmp/hf_home/cache
export TRANSFORMERS_NO_FLASH_ATTN=1
export HF_HUB_DISABLE_SYMLINKS=1
export PYTHONBREAKPOINT=0

# upgrade pip & install packages 
pip install --proxy http://hpc-proxy00.city.ac.uk:3128 --upgrade pip
pip install --proxy http://hpc-proxy00.city.ac.uk:3128 -r requirements.txt

# fix LD_LIBRARY_PATH for libffi error
export LD_LIBRARY_PATH=/opt/apps/flight/env/conda+jupyter/lib:$LD_LIBRARY_PATH

echo "Environment 'thesis_env' ready with Python $(python --version)"
python -c "import open_clip; print('OpenCLIP version:', open_clip.__version__)"

