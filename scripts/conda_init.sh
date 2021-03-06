conda create -n pl-playground
eval "$(conda shell.zsh hook)"
conda activate pl-playground

conda install -y pytorch cudatoolkit=10.1 -c pytorch
conda install -y tqdm jupyterlab -c conda-forge
pip install -U transformers pytorch-lightning ipywidgets fugashi ipadic
jupyter labextension install @jupyter-widgets/jupyterlab-manager
