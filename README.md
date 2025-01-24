1. Add conda

mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate

conda deactivate

Update Conda: conda update -n base -c default conda

Create V Env:    conda create --name TorchEnv python=3.11
Active this Env: conda activate TorchEnv

conda install pytorch torchvision torchaudio -c pytorch          // stable release
conda install pytorch torchvision torchaudio -c pytorch-nightly

2. Check
   
python -c "import torch; print(torch.__version__)"

print(torch.backends.mps.is_available())
   
3. VS Code

Install Python
VS Code: Command + Shift + P  => Search for Interpreter
Select the "~/miniconda3" env TorchEnv
