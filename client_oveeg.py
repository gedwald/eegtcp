
import socket
import struct
import numpy as np

UDP_IP = "192.168.200.240"
UDP_PORT = 10002

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

sock_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # SOCK_DGRAM
tcp_server_address = ("192.168.200.240", 10006)
sock_tcp.bind(tcp_server_address)
sock_tcp.listen(1)
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

