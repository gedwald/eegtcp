#!/usr/bin/python3

# open a socket and wait for connections; then feed the datafile to the client.  

import socket
import sys, getopt
import numpy as np
import time
from loaddata import loadeeg

def main( argv ):

  # DEFAULT values for input parametres
  # server address makes no sense; 
  server   = '192.168.200.240' # default server ('my' address)  (@TODO: shouldn't this be 'localhost') ?
  port     =  10004            # default port for this server to listening on
  fname    = "data/s01.mat"    # default comes from original laoddata 
  
  try:
    opts, args = getopt.getopt(argv,"hs:p:d:",["help","server=", "port=","data="])

  except getopt.GetoptError:
    print( 'server_oveeg.py {--help|-h } { --server|-s <server (localhost) } {--port|-p <port number>} {--data|-d <data file name> } ')
    sys.exit(2)
  for opt, arg in opts:
    if opt in ('-h',"--help"):
      print( 'server_oveeg.py {--help|-h } { --server|-s <server (localhost) } {--port|-p <port number>} {--data|-d <data file name> }')
      sys.exit()
    elif opt in ("-s","--server"):
      server = arg
    elif opt in ("-p", "--port"):
      port = int(arg)
    elif opt in ("-d", "--data"):
      fname = arg
    
  # Create a TCP/IP socket - sends data to there
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  # Bind the socket to the port
  server_address = (server, port)
  sys.stderr.write('starting up on %s port %s' % server_address)
  sock.bind(server_address)

  fs = 500
  epochsample = 32
  nchn = 16
  tlim = np.int(1.0 / epochsample * fs)

  # @TODO: how does the different fs fit with the hardcoded 512 and utime from before??
  # @TODO:  Danger !
  full_data = loadeeg( fname )

  datamat = full_data[:nchn]
  if datamat.ndim != 2:
    raise ValueError("INPUT must be 2-dim!")

  len_off_data = datamat.shape[1]
  data_buffer = np.empty((nchn, 0))

  # Listen for incoming connections
  sock.listen(1)

  while True:
    # Wait for a connection
    # TE: Depending on the host, we could also register as a service.
    # @TODO: are there multiple clients receiving the same data?
    sys.stderr.write('waiting for a connection')
    connection, client_address = sock.accept()
    try:
      sys.stderr.write('connection from', client_address)
      counter = 0
      # Read the data in small chunks and transmit it
      while True:
        mysample = datamat[:nchn, counter:counter+4].T.ravel().astype('<f4').tobytes('C')
        print( datamat[:nchn, counter:counter+4].T.ravel() )
        connection.sendall(mysample)
        counter = 0 if counter == len_off_data - 4  else counter + 4 
        time.sleep(4. / fs)
    finally:
      # Clean up the connection
      print >>sys.stderr, 'closing socket'
      connection.close()
      
if __name__ == "__main__":
   main(sys.argv[1:])
