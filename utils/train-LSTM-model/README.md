These scripts allow you to use data stored in Brain Vision files (.vhdr/.vmrk/.eeg) to train a bidirectional-LSTM network to predict R peaks on an ECG. The model can then be used in NeuXus' real-time Pulse Artifact correction (the correction code is in: neuxus/nodes/correct.py; and an example on how to use it in: examples/mri-artifact-correction/).

They are meant to be run in order:
A_format.m: 				Loads .vhdr/.vmrk/.eeg > finds the time when the fMRI acquisition started > saves .mat
B_correctGA_downsample_filter.py: 	Loads .mat > corrects the Gradient Artifact, downsamples and filters the ECG (to get the ECG processed in the same way of those which will presented to the network for prediction) > saves .mat
C_mark_QRS.m: 				Loads .mat > Helps you mark the R peaks (which will serve as the ground truth) > saves .mat
D_train.py: 				Loads .mat > Trains the LSTM network with the data and ground truth > saves the weights and biases in .pkl file.

In D_train.py, although it is not necessary to use a GPU, it's recommended since otherwise it can take very long (> 10 hours, instead of a few minutes). I trained the model in a Windows 10 computer with a NVIDIA GTX 1070 GPU. It can be tricky to install the right combination of libraries to use your GPU. Hence, I will list those I have used and worked:

1: Install CUDA, CudNN and Python
cuda_11.0.3_451.82_win10.exe
cudnn-11.0_windows-x64-v8.0.4.30.zip
Python 3.8
Later on, when training the model, I had the error: "Could not open dynamic library 'cusolver64_10.dll". So, following: "https://stackoverflow.com/questions/65608713/tensorflow-gpu-could-not-load-dynamic-library-cusolver64-10-dll-dlerror-cuso", I made the following step:
In C:\Programs\NVIDIA\...\CUDA\v11.0\bin\, I renamed cusolver64_11.dll to cusolver64_10.dll

2: Install Miniconda
Miniconda3-py39-4.9.2-windows-x86_64.exe

3: In Miniconda, create a virtual environment (I named it tf2.4), and install Python libraries:
Open Anaconda Prompt (miniconda3) and enter:
conda create --name tf2.4 python=3.8
conda activate tf2.4
pip install tensorflow==2.4
pip install wfdb==2.2.0
pip install numpy==1.19.2

4: Run D_train.py
Either run it from Anaconda Prompt inside the virtual environment (conda activate tf2.4), or in a IDE, after selecting the virtual environment. In Pycharm, this can be accomplished by: 
File>Settings>[your Project]>Python Interpreter>Add...(in the settings wheel)>Conda Environment>Existing>Interpreter>Python 3.8 (tf2.4)

PS. I provide two .mat files from after running A_format.m in data/formatted/ to allow you to run the rest of the pipeline










 