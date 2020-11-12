#!/usr/bin/python3

# open a socket and wait to read EEG data on the socket (and send it back to client) 
# (this doesn't actually do anything useful- it is pretty much an echo server)

import socket
import sys, getopt
import numpy as np

def main( argv ):

  # DEFAULT values for input parametres
  # server address makes no sense; 
  server   = '192.168.200.240' # default server ('my' address)  (@TODO: shouldn't this be 'localhost') ?
  port     =  10001            # default port for this server to listening on

  try:
    opts, args = getopt.getopt(argv,"hp:s:",["help","port=","server="])

  except getopt.GetoptError:
    print( 'server_localeeg.py {--help|-h } {--port|-p <port number>} {--server|-s <server (localhost)>}')
    sys.exit(2)
  for opt, arg in opts:
    if opt in ('-h',"--help"):
      print( 'server_localeeg.py {--help|-h } {--port|-p <port number>} {--server|-s <server (localhost)>}')
      sys.exit()
    elif opt in ("-p", "--port"):
      port = int(arg)

  # Create a TCP/IP socket
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  # Bind the socket to the port  (@TODO: Thhis should be 'localhost' right?)
  server_address = (server, port)

  sys.stderr.write("starting up on %s port %s" % server_address)
  sock.bind(server_address)

  # Listen for incoming connections
  sock.listen(1)

  while True:
    # Wait for a connection
    sys.stderr.write( "waiting for a connection" )
    connection, client_address = sock.accept()
    try:
        sys.stderr.write("connection from", client_address )

        # Receive the data in small chunks, print it and retransmit it right back
        while True:
            data = connection.recv(10000)
            true_data = np.fromstring(data[1:], sep=' ')
            print( true_data )
            if data:
                connection.sendall(data)
            else:
                break

    finally:
        connection.close()
        


if __name__ == "__main__":
   main(sys.argv[1:])
