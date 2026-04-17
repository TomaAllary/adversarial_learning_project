# Setup
### Create conda environment
conda create --name your_env

### install torch for your GPU
to check cuda version, you can type nvidia-smi in the cmd terminal.
Then go to https://pytorch.org/get-started/locally/ for the corresponding pip command

### install requirement
pip install -r requirement.txt

### Run
- Launch Visual Studio Code
- launch.json is already setup, just pick Train Agent (shooter) from the debugging list
- in pettingzoo_env/train_shooter.py, you can uncomment/comment the load/create agent lines in the main function