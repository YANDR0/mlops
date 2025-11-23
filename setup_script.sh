 #!/bin/bash

# 1. Create 2GB of Swap Memory (Fake RAM) so the ML model doesn't crash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 2. Update system and install Python
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip

# 3. Create app folder
mkdir -p /home/ubuntu/app
chmod 777 /home/ubuntu/app

# 4. Install Libraries (using --no-cache-dir to save space)
pip3 install fastapi uvicorn torch torchvision numpy --no-cache-dir