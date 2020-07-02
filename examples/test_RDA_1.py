from socket import *
from struct import *

"""
Launch with python test_RDA_1.py
"""


# Helper function for receiving whole message
def RecvData(socket, requestedSize):
    returnStream = b''
    while len(returnStream) < requestedSize:
        databytes = socket.recv(requestedSize - len(returnStream))
        if databytes == '':
            print("connection broken")
        returnStream += databytes

    return returnStream


# Helper function for splitting a raw array of
# zero terminated strings (C) into an array of python strings
def SplitString(raw):
    stringlist = []
    s = ""
    for i in range(len(raw)):
        if raw[i] != b'\x00':
            s += f'{raw[i]}'
        else:
            stringlist.append(s)
            s = ""

    return stringlist


# Helper function for extracting eeg properties from a raw data array
# read from tcpip socket
def GetProperties(rawdata):

    # Extract numerical data
    (channelCount, samplingInterval) = unpack('<Ld', rawdata[:12])

    # Extract resolutions
    resolutions = []
    for c in range(channelCount):
        index = 12 + c * 8
        restuple = unpack('<d', rawdata[index:index + 8])
        resolutions.append(restuple[0])

    # Extract channel names
    channelNames = SplitString(rawdata[12 + 8 * channelCount:])

    return (channelCount, samplingInterval, resolutions, channelNames)


def GetData(rawdata, channelCount):
    # Extract numerical data
    (block, points, markerCount) = unpack('<LLL', rawdata[:12])

    # Extract eeg data as array of floats
    data = []
    for point in range(points):
        row = []
        for j in range(channelCount):
            index = 12 + 4 * point * j
            value = unpack('<f', rawdata[index:index + 4])
            row.append(value[0])
        data.append(row)
    return (block, points, markerCount, data)


if __name__ == '__main__':
    value = input(
        "Choose port by pressing:\n1. for Recorder: 51244 \n2. for RecView: 51254\n")
    if value == 2:
        rdaport = 51254
    else:
        rdaport = 51244

    # Create a tcpip socket
    my_socket = socket(AF_INET, SOCK_STREAM)
    # Connect to recorder host via 32Bit RDA-port
    # adapt to your host, if recorder is not running on local machine
    # change port to 51234 to connect to 16Bit RDA-port
    # RECView: 51254, Recorder: 51244, use 51234 to connect with 16Bit Port
    my_socket.connect(("localhost", rdaport))

    # Flag for main loop
    finish = False

    # block counter to check overflows of tcpip buffer
    lastBlock = -1

    #### Main Loop ####
    while not finish:

        # Get message header as raw array of chars
        rawhdr = RecvData(my_socket, 24)

        # Split array into usefull information id1 to id4 are constants
        (id1, id2, id3, id4, msgsize, msgtype) = unpack('<llllLL', rawhdr)

        # Get data part of message, which is of variable size
        rawdata = RecvData(my_socket, msgsize - 24)
        print('type ', msgtype)
        print('size ', msgsize)

        # Perform action dependend on the message type
        if msgtype == 1:
            # Start message, extract eeg properties and display them
            (channelCount, samplingInterval, resolutions,
             channelNames) = GetProperties(rawdata)
            # reset block counter
            lastBlock = -1

            print("Start")
            print("Number of channels: " + str(channelCount))
            print("Sampling interval: " + str(samplingInterval))
            print("Sampling rate: " + str(1000000 / samplingInterval))
            print("Resolutions: " + str(resolutions))
            print("Channel Names: " + str(channelNames))

        elif msgtype == 4:
            # Data message, extract data and markers
            (block, points, markerCount, data) = GetData(rawdata, channelCount)
            print('point', data)

            # Check for overflow
            if lastBlock != -1 and block > lastBlock + 1:
                print("*** Overflow with " + str(block - lastBlock) + " datablocks ***")
            lastBlock = block

        elif msgtype == 3:
            # Stop message, terminate program
            print("Stop")
            finish = True

    # Close tcpip connection
    my_socket.close()
