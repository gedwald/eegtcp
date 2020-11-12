#!/usr/bin/python3

# read an EEG data file and send it off to a server/listener
#
import socket
import sys, getopt
import time
import numpy as np
from loaddata import loadeeg


def main( argv ):

  # DEFAULT values for input parametres 
  server   = '192.168.200.240' # default server (@TODO: should this be 'localhost') ?
  port     =  10001            # default port server is listening to ?
  fname    =  'data/s01.mat'   # default data file name 

  try:
    opts, args = getopt.getopt(argv,"hs:p:d:",["help","server=","port=","data"])
  except getopt.GetoptError:
    print( 'client_localeeg.py {--help|-h } {--server|-s <serverIP or name>} {--port|-p <port number>} {--data|-d <data file name>}')
    sys.exit(2)
  for opt, arg in opts:
    if opt in ('-h',"--help"):
      print( 'client_localeeg.py {--help|-h } {--server|-s <serverIP or name>} {--port|-p <port number>} {--data|-d <data file name>}')
      sys.exit()
    elif opt in ("-s", "--server"):
      server = arg 
    elif opt in ("-p", "--port"):
      port = int(arg)
    elif opt in ( "-d", "--data"):
      fname = arg

  # could add 'ds' and 'utime' as command-line parameters too, 
  # but maybe those never change ?
        
  # Create a TCP/IP socket
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

  # Connect the socket to the port where the server is listening
  server_address = (server, port)
  sys.stderr.write( 'connecting to %s port %s' % server_address )
  sock.connect(server_address)
  
  datamat = loadeeg( fname )

  if datamat.ndim != 2:
    raise ValueError("INPUT must be 2-dim array!")

  len_off_data = datamat.shape[1]

  try:
    # Send data
    counter = 0
    while True: 
      mysample = np.array_str(datamat[:, counter].ravel())
      sock.sendall(mysample)
      data = sock.recv(10000)
      if mysample != data:
        raise ValueError("Error!!!") 
      else:
        print( "True" )
      counter = 0 if counter == len_off_data - 1 else counter + 1
      time.sleep(1.0 / fs)
  finally:
    sys.stderr.write( 'closing socket' )
    sock.close()


if __name__ == "__main__":
   main(sys.argv[1:])
