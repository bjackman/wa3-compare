Android kernel perf testing tool
=======

__TODO__: Add description with glossary.

How to use
=======

Enter this repo then:

```
# Check out WA3 & devlib
git submodule update

# Install virtualenv tool
sudo apt-get install python-pip
sudo -H pip install --upgrade pip virtualenv

# Create a virtualenv
virtualenv wa3.virtualenv
source wa3.virtualenv/bin/activate

# Install WA3 & devlib
pip install -e ./wa3 ./devlib

# Set up initial WA3 config
mkdir -p ~/.workload-automation
cp ./wa-config.yaml.skel ~/.workload_automation/config.yaml

# Verify WA3 installation
wa list workloads
```

Edit `~/.workload_automation/config.yaml`: you may need to alter the acme_cape
parameters or the ADB device ID for your device.

Now you will need to populate `~/.workload_automation/dependencies` with the
.apks used for each workload. __TODO: Add instructions for getting APKs__.

__TODO: Add instructions for actually running it__.

