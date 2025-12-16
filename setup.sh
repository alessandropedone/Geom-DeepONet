#!/bin/bash

# Exit immediately if a command fails
set -e

# =========================
#      Define Colors
# =========================
GREEN='\033[0;32m'
BLUE='\033[1;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# =========================
#      Ask for Env Name
# =========================
read -p "Enter the name of your conda environment: " ENV_NAME

# =========================
#      Create Environment
# =========================
echo -e "\n${BLUE}Creating conda environment '${ENV_NAME}' with Python 3.10...${NC}"
conda create --name "${ENV_NAME}" python=3.10 -y

# =========================
#      Activate Environment
# =========================
echo -e "\n${BLUE}Activating the '${ENV_NAME}' environment...${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
echo -e "${BLUE}Environment '${ENV_NAME}' activated.${NC}"
echo ""
echo -e "${YELLOW}The current active environment is: '$(conda info --envs | grep '*' | awk '{print $1}')'.${NC}"

# =========================
#      Install Conda Packages
# =========================
echo -e "\n${GREEN}Installing core scientific packages from conda-forge...${NC}"
conda install -c conda-forge \
    fenics-dolfinx=0.9.0  \
    mpich=4.3.0 \
    pyvista=0.45.2 \
    numpy=2.2.6 \
    scipy=1.15.2 \
    matplotlib=3.10.3 \
    pandas=2.2.3 \
    tensorflow=2.18.0 \
    compilers \
    tqdm=4.67.1 \
    gmsh=4.13.1 \
    python-gmsh=4.13.1 \
    -y

# =========================
#      Install Pip Packages
# =========================
echo -e "\n${YELLOW}Installing additional Python packages with pip...${NC}"
pip install scikit-learn==1.6.1 alphashape==1.3.1

# =========================
#      Final Message
# =========================
echo -e "\n${BLUE}✅ Environment setup complete! Activate it anytime with:${NC}"
echo -e "${YELLOW}conda activate ${ENV_NAME}${NC}\n"