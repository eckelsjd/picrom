#!/bin/bash

# Run this script from the project root directory
echo "Setting up environment..."

module load python/3.11.5

# Make sure no conda environment is activated
if command -v conda deactivate &> /dev/null
then
    conda deactivate &> /dev/null
fi

# Install PDM and setup local venv
if ! command -v pdm &> /dev/null
then
    echo "Installing PDM..."
    curl -sSL https://pdm-project.org/install-pdm.py | python3 -
else
    echo "PDM already installed!"
fi

pdm install
pdm run jupyter_install  # pdm doesn't install jupyterlab correctly for some reason

# Add slurm user account info to .bashrc
if [[ -z "${SLURM_ACCOUNT}" || -z "${SLURM_MAIL}" ]]; then
  case $(whoami) in
    eckelsjd)
      export SLURM_ACCOUNT='goroda0'
      export SLURM_MAIL='eckelsjd@umich.edu'
      ;;
    *)
      export SLURM_ACCOUNT='goroda0'
      export SLURM_MAIL='eckelsjd@umich.edu'
      ;;
  esac

  echo "export SLURM_ACCOUNT=${SLURM_ACCOUNT}" >> ~/.bashrc
  echo "export SLURM_MAIL=${SLURM_MAIL}" >> ~/.bashrc
fi

echo "Environment setup complete!"
