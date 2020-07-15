from socket import (socket, AF_INET, SOCK_DGRAM)


"""
Simple Loop to receive data from a UDP client

"""

localIP     = "localhost"
localPort   = 20001
bufferSize  = 4096

def main():
    # Create a datagram socket
    my_socket = socket(family=AF_INET, type=SOCK_DGRAM)
    # Bind to address and ip
    my_socket.bind((localIP, localPort))
    print("UDP server up and listening")

    # Listen for incoming datagrams
    while True:

        bytesAddressPair = my_socket.recvfrom(bufferSize)

        message = bytesAddressPair[0]
        address = bytesAddressPair[1]

        print(f"Message from Client:{message}")
        print(f"Client IP Address:{address}")

if __name__ == '__main__':
    main()