import time
import random as rd

from pylsl import StreamInfo, StreamOutlet

# create a new StreamInfo object which shall describe our stream
info = StreamInfo('MarkerStream', 'Markers', 1, 0, 'string', 'marker1')

number_of_trials = 20
first_class = 'LEFT ARROW'
second_class = 'RIGHT ARROW'
baseline_duration = 20
wait_for_beep_duration = 2
wait_for_cue_duration = 1
display_cue_duration = 1.250
feedback_duration = 3.750
end_of_trial_min_duration = 1.5
end_of_trial_max_duration = 3.5

# now attach some meta-data
info.desc().append_child_value("Type", "Graz Motor Imagery")
config = info.desc().append_child("config")
config.append_child_value("number_of_trials", str(number_of_trials))
config.append_child_value("class1", first_class)
config.append_child_value("class2", second_class)
config.append_child_value("baseline_duration", str(baseline_duration))
config.append_child_value("wait_for_beep_duration", str(wait_for_beep_duration))
config.append_child_value("wait_for_cue_duration", str(wait_for_cue_duration))
config.append_child_value("display_cue_duration", str(display_cue_duration))
config.append_child_value("feedback_duration", str(feedback_duration))
config.append_child_value("end_of_trial_min_duration",
                          str(end_of_trial_min_duration))
config.append_child_value("end_of_trial_max_duration",
                          str(end_of_trial_max_duration))

# next make an outlet
outlet = StreamOutlet(info)

sequence = [first_class, second_class] * (number_of_trials)
for i in sequence:
    a = rd.randrange(number_of_trials * 2)
    b = rd.randrange(number_of_trials * 2)
    sequence[a], sequence[b] = sequence[b], sequence[a]

t = 0

# manages baseline
outlet.push_sample(['ExperimentStart'])
print('ExperimentStart')
time.sleep(5)

outlet.push_sample(['BaselineStart'])
print('BaselineStart')
outlet.push_sample(['Beep'])
print('Beep')
time.sleep(baseline_duration)

outlet.push_sample(['BaselineStop'])
print('BaselineStop')
outlet.push_sample(['Beep'])
print('Beep')
time.sleep(5)

# manages trials
for class_ in sequence:

    # first display cross on screen

    outlet.push_sample(['Start_Of_Trial'])
    print('Start_Of_Trial')
    outlet.push_sample(['Cross_On_Screen'])
    print('Cross_On_Screen')
    time.sleep(wait_for_beep_duration)

    # warn the user the cue is going to appear

    outlet.push_sample(['Beep'])
    print('Beep')
    time.sleep(wait_for_cue_duration)

    # display cue

    outlet.push_sample([class_])
    print(class_)
    time.sleep(display_cue_duration)

    # provide feedback

    outlet.push_sample(['Feedback_Continuous'])
    print('Feedback_Continuous')
    time.sleep(feedback_duration)

    # ends trial

    outlet.push_sample(['End_Of_Trial'])
    print('End_Of_Trial')
    time.sleep(rd.uniform(end_of_trial_min_duration, end_of_trial_max_duration))

# send end for completeness
outlet.push_sample(['End_Of_Session'])
print('End_Of_Session')
time.sleep(5)

outlet.push_sample(['ExperimentStop'])
print('ExperimentStop')
