# Detecting_subsurface_MHWs

This is the package for subsurface Marine Heatwave (MHW) detection, using sea surface temperature anomaly and sea surface height anomaly as predictors. The implemented methods include a geographically and seasonally varying coefficient (GSVC) linear regression, CNN classification (CNN_cla) learning, CNN regression (CNN_reg) learning, and ordinary least square (OLS) regression.

The package structure is outlined as follows:

GSVC Model: Estimate subsurface temperature anomaly (T') using the GSVC model (implemented in GSVC.py). We strongly encourage using MPI parallelization to run this model. Subsequently, pointwise MHWs are detected following the methodology of Hobday et al., 2016 (https://github.com/ZijieZhaoMMHW/m_mhw1.0/detect.m).

CNN_cla Model: Directly detect subsurface MHWs using the CNN_cla model, which generates a binary output (implemented in CNN_cla.py).

CNN_reg Model: Estimate T' using the CNN_reg model (implemented in CNN_cla.py) and then detect pointwise MHWs.

OLS Model: Train an OLS model at each grid to estimate T' and subsequently detect pointwise MHWs.

We will update to design an example for a specific small area. Welcome to use and modify.
