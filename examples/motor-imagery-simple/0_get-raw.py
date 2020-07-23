import numpy as np
import datetime

from neuxus.nodes import *


date = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
print (date)
features_file = '../examples/motor-imagery-simple/Epochs_raw_'+date

lsl_signal = generate.Generator('simulation', 32, 250)

#stimulation and visualization
lsl_markers = stimulator.Stimulator('../utils/stimulus/config_ov.xml') # load config
graz_vis = display.Graz(lsl_markers.output) # visualize


#temporal filtering
butter_filter = filter.ButterFilter(lsl_signal.output, 8, 12)

#left epochs
left_epochs = epoching.StimulationBasedEpoching(butter_filter.output, lsl_markers.output, 769, 0, 4)
#save features
left_features = feature.FeatureAggregator(left_epochs.output, '1')
tocsvL = store.ToCsv(left_features.output, features_file)

#right epochs
right_epochs = epoching.StimulationBasedEpoching(butter_filter.output, lsl_markers.output, 770, 0, 4)
#save features
right_features = feature.FeatureAggregator(right_epochs.output, '2')
tocsvR = store.ToCsv(right_features.output, features_file)
