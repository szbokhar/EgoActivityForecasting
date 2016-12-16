Ego-centric Capstone Project: Activity and Action Forcasting
============================================================

Author: Syed Zahir Bokhari

Supervisor: Kris Kitani

Overview
--------

Use reinforcement learning for action and activity forcasting. This directory
contains the code to use q-learning to derive a policy that can be used for activity
forecasting. The processed data files are contained in this repository.

Paper
-----

The paper that was submitted in stored inside BK-ACCV2016.zip. The final camera
ready paper is stored in the `camera ready/` directory of the zip file.

Required Packages
-----------------
argh, numpy, scipy, matplotlib, ipdb 
```% sudo apt-get install python3-numpy python3-scipy python3-matplotlib python-argh python3-ipdb```

To Run
------

```% python3 main.py basic-qlearn data/qm_densepoints.txt data/qm_hc{0}_{1}.txt 1 config_2state```
