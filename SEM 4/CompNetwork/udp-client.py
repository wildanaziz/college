from socket import *

serverName = '10.34.4.193'
serverPort = 16161
clientSocket = socket(AF_INET, SOCK_DGRAM)

message = input('[UDP] Input lowercase sentence: ')
clientSocket.sendto(message.encode(), (serverName,
serverPort))

modifiedMessage, serverAddress = clientSocket.recvfrom(2048)
print("[UDP] From Server: "+modifiedMessage.decode())
clientSocket.close()