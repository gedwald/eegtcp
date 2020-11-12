#!/usr/bin/python3

# Read a UDP data stream and send it to a TCP socket (wait for connections first)
# ( a u2t_forwarder )

import sys, getopt
import socket
import struct
import numpy as np

def main( argv ):

  # DEFAULT values for input parametres
  # ; 
  userver   = '192.168.200.240' # default address for UDP (source) to listen on
  uport     =  10002            # default port for userver to listen on
  tserver   = '192.168.200.240' # default server for TCP (consumer) to send to
  tport     =  10006            # default port for tserver 

  try:
    opts, args = getopt.getopt(argv,"hu:t:U:T:",["help","userver=","uport=","tserver=","tport="])

  except getopt.GetoptError:
    print( 'client_oveeg.py {--help|-h } {--userver|-U <server listening for UDP stream>} {--uport|-u <port to listen for UDP on>}\
{--tserver|-T <server receiving TCP stream>} {--tport|-t <port number for TCP>} ')
    sys.exit(2)
  for opt, arg in opts:
    if opt in ('-h',"--help"):
      print( 'client_oveeg.py {--help|-h } {--userver <server listening for UDP stream>} {--uport <port to listen for UDP on>}\
{--tserver <server receiving TCP stream>} {--tport <port number for TCP>} ')
      sys.exit()
    elif opt in ("-u", "--uport"):
      uport = int(arg)
    elif opt in ("-t", "--tport"):
      tport = int(arg)
    elif opt in ("-U", "--userver"):
      userver = arg
    elif opt in ("-T", "--tserver"):
      tserver = arg

  
  # receives data from here (UDP) ...
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  sock.bind((userver,uport))

  # ... sends it to there ... (TCP) ...
  tcp_server_address = (tserver,tport)
  sock_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # SOCK_DGRAM
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

if __name__ == "__main__":
   main(sys.argv[1:])
