import socket
 
print("Host Name ",socket.gethostname())
 
print("IP Address(Internal) : ",socket.gethostbyname(socket.gethostname()))
 
print("IP Address(External) : ",socket.gethostbyname(socket.getfqdn()))
print(socket.gethostbyname(socket.gethostname()))

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
print(s.getsockname()[0]) 