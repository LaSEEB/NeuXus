from neuxus.nodes import *

data_path = r'file.vhdr'
weight_path = r'weights-input-500.pkl'

# signal = io.RdaReceive(rdaport=51244)
signal = read.Reader(data_path)

signal_ga = correct.GA(signal.output, start_marker='Response/R128', marker_input_port=signal.marker_output)  # 'Response/R128' is the marker of the start of every MRI volume (in case the data is read from a Brain Vision file; in case it's streamed by Brain Vision Recorder, it is 'R128')
signal_dw = filter.DownSample(signal_ga.output, int(5000 / 250))
signal_pa = correct.PA(signal_ds.output, weight_path, marker_input_port=signal_ga.marker_output, start_marker='Start of GA subtraction', stride=50)

signal_m_lsl = io.LslSend(signal.marker_output, 'marker', type='Markers', format='string')
signal_pa_lsl = io.LslSend(signal_pa.output, 'signal_pa', type='EEG')
signal_pa_m_lsl = io.LslSend(signal_pa.marker_output, 'marker_pa', type='Markers', format='string')
