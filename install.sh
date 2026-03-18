#!/bin/bash
set -e
echo "Stopping services"
systemctl disable thermal-recorder-py
systemctl stop thermal-recorder-py
systemctl disable thermal-classifier
systemctl disable thermal-postprocess
systemctl stop thermal-classifier
systemctl stop thermal-classifier
echo "Installing Medium Power PY"
cd /home/pi/
git clone https://github.com/gferraro/MediumPower.git
cd MediumPower
/home/pi/.venv/classifier/bin/pip install -r requirements.txt
echo "Install services to systemd"
cp thermal-medium-power.service /etc/systemd/system/
systemctl enable thermal-medium-power
daemon-reload


cd ..
echo "Setting config.toml"
cp /etc/cacophony/config.toml /home/pi/config.toml.bak
sed -i '/use-low-power-mode = true/a instant-classify = true' /etc/cacophony/config.toml

echo "Installing tc2-agent"
wget https://github.com/TheCacophonyProject/tc2-agent/releases/download/v0.8.4/tc2-agent_0.8.4_arm64.deb
sudo dpkg -i tc2-agent_0.8.4_arm64.deb

echo "Starting thermal medium power"
systemctl start thermal-medium-power
systemctl status thermal-medium-power