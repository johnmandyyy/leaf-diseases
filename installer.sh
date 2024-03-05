#!/bin/bash

# Backup your system (optional but recommended)
sudo apt update
sudo apt upgrade

# Install Python 3.8.5 using the deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.8

# Set the alternatives to configure the default Python version
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Choose the default Python version
sudo update-alternatives --config python3

# Verify the change
python3 --version
