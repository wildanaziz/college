from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5
from Crypto.Random import get_random_bytes
import base64

encrypted_base64_input = input("Masukkan pesan Base64: ")

with open("private.key", "rb") as f:
    private_key = RSA.import_key(f.read())

cipher = PKCS1_v1_5.new(private_key)

try:
    encrypted_bytes = base64.b64decode(encrypted_base64_input)
except base64.binascii.Error:
    print("Input bukan Base64 yang valid.")
    exit(1)

sentinel = get_random_bytes(32)
decrypted = cipher.decrypt(encrypted_bytes, sentinel)

try:
    print("Pesan terdekripsi:", decrypted.decode())
except UnicodeDecodeError:
    print("Dekripsi gagal atau data bukan teks (mungkin biner).")
