# eegtcp
Jiachen XU's code for EEG communications - first import

In subdir 'workdir' : files I have removed from the main directory
as they seemed to be only useful in the development- templates eg.

Worked the 'local' pair by adding a getopt-list of command line
parametres.  the client just reads the (possibly hardcoded default)
file and sends its data across to the server.
The server only receives it and prints it out, and sends it back.
(The back-transport is used to verify the transmisison.)
Pretty useless pair of files (~'scp')



