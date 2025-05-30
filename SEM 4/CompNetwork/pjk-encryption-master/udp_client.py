import socket
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5

with open("cert.pem", "rb") as f:
    server_pub_key = RSA.import_key(f.read())

encrypt_cipher = PKCS1_v1_5.new(server_pub_key)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('10.34.4.214', 4445)

print("UDP Client Started")

msg = input("Pesan: ")
if not msg:
    print("No input provided. Exiting.")
else:
    encrypted = encrypt_cipher.encrypt(msg.encode())
    sock.sendto(encrypted, server_address)

    data, _ = sock.recvfrom(4096)
    decrypted_b64 = data.decode()

    print(f"Data terenkripsi (Base64):\n{decrypted_b64}\n")
