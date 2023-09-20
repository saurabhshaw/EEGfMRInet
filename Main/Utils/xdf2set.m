function xdf2set(xdf_datafile,include_chans,chanlocs_file,dataset_name,output_path)

EEG = pop_loadxdf(xdf_datafile , 'streamtype', 'EEG', 'exclude_markerstreams', {});
EEG = pop_select( EEG, 'channel',include_chans);
EEG = pop_chanedit(EEG, 'load',{chanlocs_file 'filetype' 'autodetect'}); EEG = eeg_checkset( EEG );
EEG.setname = [dataset_name]; EEG = eeg_checkset( EEG );
pop_saveset(EEG, 'filename',[dataset_name '.set'],'filepath',output_path);