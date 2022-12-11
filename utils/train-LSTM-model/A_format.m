%% Add paths
clear, clc, close all
addpath('lib/matlab/')
addpath('C:/Users/guta_/Desktop/Data Analysis/Libraries/eeglab2021.0');  % EEGLAB path. Necessary if data is in Brain Vision format
varsbefore = who; eeglab; varsnew = []; varsnew = setdiff(who, varsbefore); clear(varsnew{:})

%% Set
lfol = 'data/unformatted/';
sfol = 'data/formatted/';
lfil = 'sub-patient005_ses-interictal_task-eegcalibration_eeg.vhdr';
sfil = 'sub-patient005_ses-interictal_task-eegcalibration_chan-ecg.mat';

%% Load
EEG = pop_loadbv(lfol, strcat(lfil), [], []);  % In case data is in Brain Vision format (.vhdr,.vmrk,.eeg)
EEG.data = double(EEG.data);

%% Mark fMRI volumes (optional)
% If there is no marker marking the start of the fMRI acquisition, generate it (+ for every other fMRI volume)
% The first marker will serve as an indication to the GA correction (next step) to start correcting from there. 
% If there is, this step can be skipped
TR = 1.26;
thr = 0.018;
grad_thr = [100,200];
expected_vols = 268;  % The number of fMRI volume markers you expect to find
tolerance = 10;
for j = 1:numel(thr)
    for k = 1:numel(grad_thr)
        [~, EEG, vols] = find_volume_markers(EEG, TR, thr(j), grad_thr(k)); % sets a Scan Start marker for each volume
        if abs(expected_vols - numel(vols)) < tolerance
            break
        end
    end
end

% Check if an unexpected number of volumes was detected
if abs(expected_vols - numel(vols)) > tolerance
    fprintf('Found %d vols but expected %d in %s! Do not continue!\n',numel(vols),expected_vols,lfil)
    return
end

%% Trim to fMRI acquisition (optional)
% Remove a possible incomplete volume at the end
if vols(end)+TR*EEG.srate-1 > size(EEG.data, 2)
    vols(end) = [];
end

% Trim
EEG = trim(EEG,'latency',[vols(1), vols(end)+TR*EEG.srate-1]);
vols = vols-vols(1)+1;
EEG.volmarkers = vols;

%% Plot volume (optional)
figure
eeg_chn = find(ismember(upper({EEG.chanlocs(:).labels}),{'C3'}));
plot(EEG.times/1000, EEG.data(eeg_chn, :))
hold on
plot(EEG.times(EEG.volmarkers)/1000, EEG.data(eeg_chn,EEG.volmarkers), 'm*')
title(lfil, 'Interpreter', 'none')

%% Select ECG (the only channel needed to ultimately train the LSTM R peak detector)
ecg_chn = find(ismember(upper({EEG.chanlocs(:).labels}),{'ECG','EKG'}));
EEG = pop_select(EEG,'channel',ecg_chn);

%% Save
save(strcat(sfol, sfil), 'EEG');
