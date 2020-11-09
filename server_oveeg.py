import socket
import sys
import numpy as np
import time
from loaddata import loadeeg

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # SOCK_DGRAM
# Bind the socket to the port
# server_address = ('192.168.200.240', 10002)
server_address = ('192.168.200.240', 10004)
print >>sys.stderr, 'starting up on %s port %s' % server_address
sock.bind(server_address)
fs = 500
epochsample = 32
nchn = 16
tlim = np.int(1.0 / epochsample * fs)

data_buffer = np.empty((nchn, 0))
# Listen for incoming connections
sock.listen(1)
full_data = loadeeg()
datamat = full_data[:nchn]



"""
timeBuffer = 1024
time_ = np.arange(0, timeBuffer/10., 0.1)
datamat = np.zeros((nchn, timeBuffer))
for rowIndex in range(nchn):
	# datamat[rowIndex, :] = np.float32(np.sin( 2.*np.pi*(rowIndex+1.)*timeBuffer))
	datamat[rowIndex, :] = np.float32(np.sin(2. * np.pi* time_ * (rowIndex + 1.)/ 20.))
        # datamat[rowIndex, :512] = 0 # np.float32(0)
	# datamat[rowIndex, 512:] = 1 # np.float32(1)  ## np.sin( 2.*np.pi*(rowIndex+1.)*timeBuffer )
# datamat[-1, :512] = np.uint8(0)
# datamat[-1, 512:] = np.uint8(1)
# datamat[-2, :512] = np.uint8(0)
# datamat[-2, 512:] = np.uint8(1)
# datamat[-3, :512] = np.uint8(0)
# datamat[-3, 512:] = np.uint8(1)
"""

if datamat.ndim != 2:
    raise ValueError("INPUT must be 2-dim!")

len_off_data = datamat.shape[1]

while True:
    # Wait for a connection
    print >>sys.stderr, 'waiting for a connection'
    connection, client_address = sock.accept()
    try:
        print >>sys.stderr, 'connection from', client_address
        counter = 0
        # Receive the data in small chunks and retransmit it
        while True:
            mysample = datamat[:nchn, counter:counter+4].T.ravel().astype('<f4').tobytes('C')
            print datamat[:nchn, counter:counter+4].T.ravel()

            connection.sendall(mysample)
            print >>sys.stderr, 'sending ' # %datamat[:nchn, counter].ravel() # "" %mysample
            counter = 0 if counter == len_off_data - 4  else counter + 4 
            time.sleep(4. / fs)
    finally:
        # Clean up the connection
        connection.close()
      
"""


# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # SOCK_DGRAM
# Bind the socket to the port
# server_address = ('192.168.200.240', 10002)
server_address = ('127.0.0.1', 10002)
print >>sys.stderr, 'starting up on %s port %s' % server_address

fs = 500
epochsample = 32
nchn = 32
tlim = np.int(1.0 / epochsample * fs)


UDP_IP = "127.0.0.1"
UDP_PORT = 10002

data_buffer = np.empty((nchn, 0))
# Listen for incoming connections
full_data = loadeeg()
datamat = full_data[:nchn]

if datamat.ndim != 2:
    raise ValueError("INPUT must be 2-dim!")

len_off_data = datamat.shape[1]

while True:

    try:
        counter = 0
        # Receive the data in small chunks and retransmit it
        while True:
            mysample = datamat[:nchn, counter:counter+4].T.ravel().astype('<f4').tobytes('C')
            # print datamat[:nchn, counter:counter+4].T.ravel()
            print mysample

            sock.sendto(mysample, (UDP_IP, UDP_PORT))
            print >>sys.stderr, 'sending ' # %datamat[:nchn, counter].ravel() # "" %mysample
            counter = 0 if counter == len_off_data - 4  else counter + 4 
            time.sleep(4. / fs)
    finally:
        pass
        

"""