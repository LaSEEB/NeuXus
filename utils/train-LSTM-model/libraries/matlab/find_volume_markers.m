function [fig, EEGNEW, g_event_latency]  = find_volume_markers(EEG, TR, thr, grad_thr)

% highPassFilter = 1;  % Hz
% data = pop_eegfiltnew(EEG, highPassFilter, []);

%% 
data = EEG;
new_data = data;
fs = EEG.srate;
% grad_thr = 100; % gradient trigger 200 uV/ms
% grad_thr = 200; % gradient trigger 200 uV/ms
nr_slices = 20; % 20 slices per volume
TRslice = (TR/nr_slices)*EEG.srate; % number of points of a slice
w = (1/EEG.srate)*1000; % time between two points in ms

% use average of all channels (except ECG, the last channel):
n_chan = data.nbchan; 
l_chan = 1:n_chan;
template = ismember(l_chan, 1:n_chan-1); 
av_chan = (template * [data.data]  /sum(template));

grad_ids = find(abs(gradient(av_chan, w)) > grad_thr); % selects only the points
                                                  % above gradient threshold

% figure;
% plot(0:1/fs:(length(av_chan)-1)/fs, av_chan)
% hold on
% plot(grad_ids./fs, ones(1, length(grad_ids)), '*')
% xlabel('Time [s]')
% ylabel('Amplitude [uV]')


diff_ids = diff(grad_ids); % distance in points between the selected indexes
val = unique(diff_ids);
cnt = histc(diff_ids, val);
counting = [val; cnt];


thr_diff = diff_ids(diff_ids <thr*fs); % distance between consecutive
                                       % indexes has to be lower than
                                       % TRslice. In this case, a
                                       % stricter value, TRslice/3 was
                                       % used

                                           
m = max(thr_diff);
ind = unique(thr_diff(thr_diff >= m - 10)); % from those, select only the 
                                           % ones for which the distance 
                                           % does not differ from the
                                           % maximum more than 10 points
                                        
x = ismember(diff_ids, ind); 

ids = grad_ids(find(x)+1); 

% take out last point if there is no artifact after it, by comparing the
% maximum signal amplitudes in a window after the marker with the period 
% before the marker
try 
    
last = ids(end);
w_before = last-TR/2*EEG.srate:last-1; % window before last point
w_after = last+(TR/2)*EEG.srate+1:last+TR*EEG.srate;% window TR/2 after last point
                                                    %(the window does not start 
                                                    %right after the last marker
                                                    %because there might be brief 
                                                    %large amplitude oscillations)

if max(data.data(1, w_after)) < 0.5 * max(data.data(1, w_before))
    ids = ids(1:end-1);
end

catch
end

% CONFIRM TIMINGS!
timings = diff(ids); 

% exclude points that do not differ approximately TR from their neighbour
if ~all(abs(timings-TR*fs) < 100) 
    ids = [ids(abs(timings-TR*fs) < 100), ids(end)];
end

vols = length(ids)-2;

% Final plot with volumes 
% fig = figure;
% plot(0:1/fs:(length(av_chan)-1)/fs, av_chan)
% hold on
% plot(ids(3:end)./fs, av_chan(ids(3:end)), 'r*') % the first 2 markers belong 
%                                                 % to the dummy period
% xlabel('Time [s]')
% ylabel('Amplitude [uV]')
% title(['Number of volumes: ', num2str(vols)])
% legend('Average EEG', 'Volumes')

%% Add markers

g_event_latency = ids(3:end); % the first 2 markers belong to the dummy period
n_events = length(g_event_latency);

if isempty(data.event) 
    event_latency = num2cell(g_event_latency);
else
    event_latency = num2cell([g_event_latency, data.event.latency]);
end

% 'type'. All the markers have the same type: 'Scan Start' 
event_type2 = strsplit(repmat('Scan Start,',1,n_events),',');
event_type1 = event_type2(1:n_events);
event_type = [event_type1, data.event.type];

% duration set to 1
if isfield(data.event, 'duration')
    event_duration = num2cell([ones(1, n_events),data.event.duration]);
else 
    event_duration = num2cell([ones(1, n_events), ones(1, length(data.event))]);
end

field1 = 'type'; value1 = event_type;
field2 = 'latency'; value2 = event_latency;
field3 = 'duration'; value3 = event_duration;

own_event = struct(field1, value1, field2, value2, field3, value3);

[~,index] = sortrows({own_event.latency}'); 
own_event = own_event(index);

 % The field 'urevent' is added to the structure
event_id = num2cell(1:length(own_event));
[own_event(:).urevent] = deal(event_id{:});

% Acording with the eeglab files structures, It is needed to create the
% struct urevent
own_urevent = rmfield(own_event,'urevent');

new_data.event = own_event;
new_data.urevent = own_urevent;

EEGNEW = new_data;

fig = 0;
end