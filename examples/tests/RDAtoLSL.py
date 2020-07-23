from neuxus.nodes import *


# receive RDA from Recorder or Recview
#lsl_markers = io.LslReceive('type', 'Markers', data_type='marker')
rda_reception = io.RdaReceive(rdaport=51244, host="192.168.1.132") #Recorder:51244 RecView:51254


#send via LSL
lsl_send = io.LslSend(rda_reception.output, 'rda_reception', 'EEG')


#display
#disp = display.Plot(rda_reception.output)
