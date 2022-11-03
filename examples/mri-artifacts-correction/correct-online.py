from neuxus.nodes import *

data_path = r'file.vhdr'
weight_path = r'weights.pkl'

signal = read.Reader(data_path)

signal_ga = correct.GA(signal.output, start_marker='Stimulus/S  1', marker_input_port=signal.marker_output)
signal_dw = filter.DownSample(signal_ga.output, int(5000 / 250))
signal_dw_ecg = select.ChannelSelector(signal_dw.output, 'name', ['ECG'])
signal_dw_ecg_fi = filter.ButterFilter(signal_dw_ecg.output,  0.5, 30)
signal_pa = correct.PA(signal_dw.output, weight_path, 500, 50, start_marker='Start of GA correction', marker_input_port=signal_ga.marker_output)

signal_pa_lsl = io.LslSend(signal_pa.output, 'signal_pa', type='signal')
signal_r_m_lsl = io.LslSend(signal_pa.marker_output_r, 'marker_r', type='marker', format='string')
signal_ga_m_lsl = io.LslSend(signal_ga.marker_output, 'marker_ga', type='marker', format='string')
signal_pa_m_lsl = io.LslSend(signal_pa.marker_output_pa, 'marker_pa', type='marker', format='string')
