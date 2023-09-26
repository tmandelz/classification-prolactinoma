befehle in CMD, CSCS Jupyterhub oder VSCODE 
{platzhalter} sind plathhalter und sollen ersetzt werden
kommentare sind mit # markiert und sollen nicht benützt werden

in CMD:
ssh {user}@ela.cscs.ch  
#(cscs credentials)
ssh daint 
#(cscs credentials)
module load cray-python
module load daint-gpu
module load jupyter-utils
mkdir ccv1
cd ccv1
git clone https://gitlab.fhnw.ch/thomas.mandelz/tierli_ahluege.git 
#(gitlab credentials)
cd tierli_ahluege
pipenv install
pipenv install ipykernel
pipenv shell
kernel-create -n ccv1


in CSCS Jupyterhub:
https://{username}.jupyter.cscs.ch/hub/token generieren
token kopieren bsp: 02be80350ca324d4b7a07e5191c2d7d6

jhub server starten


in VSCODE: 
jupyterhub extension in vs code installieren
Kernel in Jupyter Notebook wählen -> Existing Jupyter Servers -> Enter the URL of the running Jupyter server
URL zum eingeben: https://{username}.jupyter.cscs.ch/user/{username}/?token={Token}

BSP:https://tmandelz.jupyter.cscs.ch/user/tmandelz/?token=02be80350ca324d4b7a07e5191c2d7d6

choose ccv1 in jupyternotebook

# Git-lfs in CSCS

ssh {user}daint.cscs.ch
module load daint-gpu
module load EasyBuild-custom/cscs
eb git-lfs-3.2.0.eb --try-software-version=3.3.0 -r

module avail git-lfs

module load git-lfs

cd ccv1/tierli_ahluege
git lfs install

## to update use

git lfs pull
