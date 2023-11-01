# EEGfMRInet
#

This package greatly improves the accessibility of brain network-based measures (conventionally assessed using fMRI) by providing a framework predict brain network activation using EEG data. 
To accompish this, the package learns EEG features of large-scale brain networks using simultaneous EEG-fMRI data, with the end goal of using the learned EEG features to predict brain network activation using only EEG data. A pretrained model is also provided that can be directly applied on EEG-only data.

Further details about the methodology and performance can be found in the following paper:
Shaw, S. B., McKinnon, M. C., Heisz, J. J., Harrison, A. H., Connolly, J. F., & Becker, S. (2021). Tracking the brain’s intrinsic connectivity networks in EEG. bioRxiv, 2021-06. doi: https://doi.org/10.1101/2021.06.18.449078

This framework has also been used in an EEG-only setting to track changes in brain networks after a neurofeedback/aerobic exercise protocol:
Shaw, S. B., Levy, Y., Mizzi, A., Herman, G., McKinnon, M. C., Heisz, J. J., & Becker, S. (2022). Combined Aerobic Exercise and Neurofeedback Lead to Improved Task-Relevant Intrinsic Network Synchrony. Frontiers in Human Neuroscience, 16, 838614.

# How to Run:
Clone the repository and run the process_ExampleEEG.m script within the ExampleRun folder to see the framework preprocess and compute the features from an example EEG file collected from a participant while they had their eyes closed.

To use the framework with your EEG dataset, duplicate the process_ExampleEEG.m script and modify the parameters to match your dataset.

The framework utilizes EEGLAB toolbox to load EEG files and save the results, hence filetypes that can be read by EEGLAB are compatible with this framework.

