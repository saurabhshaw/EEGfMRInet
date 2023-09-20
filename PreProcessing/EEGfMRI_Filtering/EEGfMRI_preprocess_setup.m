function [EEGfMRI_preprocess_param] = EEGfMRI_preprocess_setup()

EEGfMRI_preprocess_param = []; % Setting the parameters for the FASTR GA Filtering algorithm 
EEGfMRI_preprocess_param.lpf = 70; % Low Pass Filter Cut-off
EEGfMRI_preprocess_param.L = 10; % Interpolation folds
EEGfMRI_preprocess_param.window = 30; % Averaging window
EEGfMRI_preprocess_param.Trigs = []; % Trigger Vector
EEGfMRI_preprocess_param.strig = 1; % Slice Triggers or Volume Triggers
EEGfMRI_preprocess_param.anc_chk = 1; % Run ANC
EEGfMRI_preprocess_param.tc_chk = 0; % Run Slice timing correction
EEGfMRI_preprocess_param.rel_pos = 0.03; % relative position of slice trig from beginning of slice acq
                             % 0 for exact start -> 1 for exact end
                             % default = 0.03;
% EEGfMRI_preprocess_param.exclude_chan = [scan_parameters.ECG_channel]; % Channels not to perform OBS  on.
EEGfMRI_preprocess_param.num_PC = 'auto'; % Number of PCs to use in OBS.

EEGfMRI_preprocess_param.low_bp_filt = 0.1; EEGfMRI_preprocess_param.high_bp_filt = 70;

EEGfMRI_preprocess_param.parallel = 'cpu'; % Can be cpu or gpu(under development)
EEGfMRI_preprocess_param.EEGLAB_preprocess_BPfilter = 0;
EEGfMRI_preprocess_param.ICA_data_select = 1;
EEGfMRI_preprocess_param.ICA_data_select_range = 20;