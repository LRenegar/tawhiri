#!/bin/bash
# Tawhiri deployment example
# Run as root/sudo
echo This is a deployment EXAMPLE only\; it is intended as a reference.
echo Running this script is entirely at your own risk.
echo It may break existing configurations.
echo Please type \"I accept\" to proceed\, or enter to exit.
read userinput
if [[ $userinput != $"I accept" ]]
then
    exit 0
fi

cd ../..
cp --recursive ./tawhiri /srv/tawhiri
cd /srv/tawhiri
apt update
apt install supervisor python3-dev python3-venv virtualenv imagemagick build-essential libgrib-api-dev libevent-dev libpng-dev libeccodes-dev
useradd -r tawhiri
usermod -L tawhiri
chown --recursive tawhiri:tawhiri .
touch /srv/ruaumoko-dataset
chown tawhiri:tawhiri /srv/ruaumoko-dataset
su tawhiri
python3 -m venv venv
source venv/bin/activate
pip install numpy wheel
pip install pyproj
pip install pygrib gevent
pip install -r requirements.txt
python setup.py build_ext --inplace
ruaumoko-download  # TODO may need to adust imagemagick resource settings for this to work
exit

mkdir /srv/tawhiri-datasets
chown tawhiri:tawhiri /srv/tawhiri-datasets
mkdir /var/log/tawhiri
chown tawhiri:tawhiri /var/log/tawhiri
cp deploy/supervisord.conf /etc/supervisor/conf.d/tawhiri.conf
cp deploy/logrotate.conf /etc/logrotate.d/tawhiri

supervisorctl reread
supervisorctl restart all

# Uncomment lines to deploy to nginx
#apt install nginx python-certbot-nginx
#cp deploy/nginx.conf /etc/nginx/conf.d/tawhiri.conf
 