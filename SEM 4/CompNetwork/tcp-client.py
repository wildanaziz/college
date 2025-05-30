from socket import *

serverName = '10.34.4.193'
serverPort = 15151
clientSocket = socket(AF_INET, SOCK_STREAM)
clientSocket.connect((serverName, serverPort))

sentence = input('[TCP] Input lowercase sentence: ')
clientSocket.send(sentence.encode())
modifiedSentence = clientSocket.recv(1024)
print('[TCP] From Server:', modifiedSentence.decode())

clientSocket.close()