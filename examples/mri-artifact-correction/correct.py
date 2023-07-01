from neuxus.nodes import *

data_path = r'file.vhdr'
weight_path = r'weights-input-500.pkl'

# signal = io.RdaReceive(rdaport=51244)
signal = read.Reader(data_path)

signal_ga = correct.GA(signal.output, marker_input_port=signal.marker_output, start_marker='Response/R128')  # 'Response/R128' is the marker of the start of every MRI volume (in case the data is read from a Brain Vision file; in case it's streamed by Brain Vision Recorder, it is 'R128')
signal_ds = filter.DownSample(signal_ga.output, int(5000 / 250))
signal_ds_ecg = select.ChannelSelector(signal_ds.output, 'name', ['ECG'])
signal_ds_ecg_fi = filter.ButterFilter(signal_ds_ecg.output,  0.5, 30)
signal_ds_fi = select.ChannelUpdater(signal_ds.output, signal_ds_ecg_fi.output)
signal_pa = correct.PA(signal_ds_fi.output, weight_path, marker_input_port=signal_ga.marker_output, start_marker='Start of GA subtraction', stride=50)

signal_m_lsl = io.LslSend(signal.marker_output, 'marker', type='Markers', format='string')
signal_pa_lsl = io.LslSend(signal_pa.output, 'signal_pa', type='EEG')
signal_pa_m_lsl = io.LslSend(signal_pa.marker_output, 'marker_pa', type='Markers', format='string')
