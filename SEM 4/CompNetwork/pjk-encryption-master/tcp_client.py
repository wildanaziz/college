import socket
import ssl

HOST = '10.34.4.214'
PORT = 4444

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE
context.set_ciphers("RSA")

with context.wrap_socket(sock, server_hostname=HOST) as ssock:
    ssock.connect((HOST, PORT))
    data = ssock.recv(1024)
    print(f"Data dari server: {data.decode()}")
