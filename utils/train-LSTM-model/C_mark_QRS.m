%% Add paths
clear, clc, close all
% Download and insert deepQRS and interactiveQRS from here:
% https://github.com/LaSEEB/deepQRS
% https://github.com/LaSEEB/interactiveQRS
% And put them in here:
addpath(genpath('lib/matlab/'))
addpath('C:/Users/guta_/Desktop/Data Analysis/Libraries/eeglab2021.0');  % EEGLAB path. Necessary if data is in Brain Vision format
varsbefore = who; eeglab; varsnew = []; varsnew = setdiff(who, varsbefore); clear(varsnew{:})

%% Set
lfol = 'data/corrected/';
sfol = 'data/detected/';
lfil = 'sub-patient005_ses-interictal_task-eegcalibration_chan-ecg_cor-fiNeXoff.mat';
sfil = 'sub-patient005_ses-interictal_task-eegcalibration_chan-ecg_cor-fiNeXoff_det-qrs.mat';
mfol = 'lib/matlab/deepQRS/model/';
mfil = 'weights.mat';

%% Load
load(strcat(lfol, lfil), 'EEG');
ecg_chn = find(ismember(upper({EEG.chanlocs(:).labels}),{'ECG','EKG'}));

%% Make a preliminary automatic R peak detection to help with the interactive detection
% Detect R peaks using automatic method (optional)
% freq_band = [4 45]; % [4 45 Hz] to increase QRS detection accuracy (Abreu et al., 201
% reverse = 1;
% [~, ~, starter_r_latencies, ~] = ecgPeakDetection_v4(EEG.data(ecg_chn, :), EEG.srate, freq_band, reverse);

% Detect R peaks using deep learning method (recommended)
W = load(strcat(mfol,mfil));
starter_r_latencies = deepQRS(EEG.data(ecg_chn, :),W);

%% Detect R peaks using interactive method
mark = true;
while mark
    EEG = interactiveQRS(EEG,starter_r_latencies);  % The interactively marked R peaks will be stored in EEG.event as 'QRSi'
    mark = ~input('Save interactively QRS-detected EEG? (1/0)\n');
    starter_r_latencies = [EEG.event(strcmp({EEG.event(:).type}, 'QRSi')).latency];
end      

fprintf('Calculated average heart rate: %d\n', round((60*numel(starter_r_latencies))/(EEG.pnts/EEG.srate)))

%% Save
save(strcat(sfol, sfil), 'EEG');
