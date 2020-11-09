

import socket
import struct
import numpy as np

UDP_IP = "192.168.200.240"
# UDP_IP = "127.0.0.1"
UDP_PORT = 10002



sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)# Internet
sock.bind((UDP_IP, UDP_PORT))

sock_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # SOCK_DGRAM
tcp_server_address = ("192.168.200.240", 10006)
sock_tcp.bind(tcp_server_address)
sock_tcp.listen(1)
"""
	print len(data)
	print data.encode("hex")
	print struct.unpack('f', data[4:8])
"""
n_chn = 16
np_samples = np.zeros([4, n_chn])
while True: 
	connection, client_address = sock_tcp.accept() 
	try:
		while True:
			cnt = 0
			while  cnt < 4:
				data, addr = sock.recvfrom(2048) # buffer size is 1024 bytes
				samples = data[28:]
				np_samples[cnt] = np.asarray([int.from_bytes(samples[x:x+3], byteorder='big', signed=True) for x in range(0, len(samples),3 )])
				cnt += 1
			mysample = np_samples.ravel().astype('<i4').tobytes('C')
			print(np_samples)
			connection.sendall(mysample)
			
	finally:
		connection.close



	# print "received message:", struct.unpack(">f", data)






"""

import socket
import struct
import numpy as np

UDP_IP = "192.168.200.240"
# UDP_IP = "127.0.0.1"
UDP_PORT = 10002



sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)# Internet
sock.bind((UDP_IP, UDP_PORT))

sock_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # SOCK_DGRAM
tcp_server_address = ("192.168.200.240", 10004)
sock_tcp.bind(tcp_server_address)
sock_tcp.listen(1)

	print len(data)
	print data.encode("hex")
	print struct.unpack('f', data[4:8])

while True: 
	connection, client_address = sock_tcp.accept() 
	try:
		while True:
			data, addr = sock.recvfrom(2048) # buffer size is 1024 bytes
			samples = data[28:]

			np_samples = np.asarray([int.from_bytes(samples[x:x+3], byteorder='big', signed=True) for x in range(0, len(samples),3 )])
			mysample = np_samples.astype('<i4').tobytes('C')
			print(np_samples)
			connection.sendall(mysample)
			

	finally:
		connection.close



# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # SOCK_DGRAM
# Bind the socket to the port
# server_address = ('192.168.200.240', 10002)
server_address = ('127.0.0.1', 10002)
print >>sys.stderr, 'starting up on %s port %s' % server_address
sock.bind(server_address)
fs = 500
epochsample = 32
nchn = 32
tlim = np.int(1.0 / epochsample * fs)

data_buffer = np.empty((nchn, 0))
# Listen for incoming connections
sock.listen(1)
full_data = loadeeg()
datamat = full_data[:nchn]




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
        pass
















                               0000-009e-3bd2-fffd

02 00 0000 0000-040b 0001 0005 0000-0000-0000-1437 0000-0000-009d-edb1 fff-d8b fff-d9a fff-d90 fff-d76 fff-d58
02 00 0000 0000-040c 0001 0005 0000-0000-0000-143c 0000-0000-009e-14c2 fffd 3af-ffd 2cf-ffd-3df ffd69fffd92

02 00 0000 0000-040d 0001 0005 0000-0000-0000-1441 0000-0000-009e-3bd2 fffd a0f-ffd a5f-ffd-9ef ffd7afffd5b
020000000000040d 00010005000000000000144100000000009e3bd2   fffda0fffda5fffd9efffd7afffd5b

"""