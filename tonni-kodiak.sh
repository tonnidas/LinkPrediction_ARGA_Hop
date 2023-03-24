# Navigate to current folder
cd $PBS_O_WORKDIR
pwd

# Activate Conda environment
module load use.own
conda -V
eval "$(conda shell.bash hook)"
conda activate arga
python -V

# Run python script
cd ARGA/arga
python make_hopped_info.py -- --dataname=citeseer --hop=2
