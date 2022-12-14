function EEG = trim(EEG,unit,lims)
% E.g.:
% EEG = trim(EEG,'time',[20,60]);
% EEG = trim(EEG,'latency',[5000,15000]);
% EEG = trim(EEG,'marker',{'S  1','S 12'});

%% Find limits
switch unit
    case 'time'
        mask = EEG.times/1000 >= lims(1) & EEG.times/1000 <= lims(2);
        id1 = find(mask, 1);
        id2 = find(mask, 1, 'last');
    case 'latency'
        id1 = lims(1);
        id2 = lims(2);
    case 'marker'
        id1 = EEG.event(find(strcmp({EEG.event(:).type},lims{1}),1,'first')).latency;
        evs = find(strcmp({EEG.event(:).type},lims{2}));
        ev2 = evs(end-1); % TO GET SECOND-TO-LAST R128, (and not the last)
        id2 = EEG.event(ev2).latency;
end

%% Trim
EEG.data = EEG.data(:, id1:id2);
EEG.times = EEG.times(:, id1:id2);
EEG.xmin = EEG.times(1)/1000;
EEG.xmax = EEG.times(end)/1000;
EEG.pnts  = size(EEG.data,2);

if ~isempty(EEG.event)
    mask = [EEG.event(:).latency] >= id1 & [EEG.event(:).latency] <= id2;
    EEG.event = EEG.event(mask);
    new_latencies = [EEG.event(:).latency] - id1 + 1;
    
    for i = 1:numel(EEG.event)
        EEG.event(i).latency = new_latencies(i);
        EEG.event(i).urevent = i;
    end
end

%% Tare
EEG.times = EEG.times - EEG.times(1);
EEG.xmin = EEG.times(1)/1000;
EEG.xmax = EEG.times(end)/1000;

end
