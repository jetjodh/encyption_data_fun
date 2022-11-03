import random
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.padding import PKCS7
from cryptography.hazmat.primitives.asymmetric import rsa, dsa, ec, dh, padding
import os
import pandas as pd

# dictionary of cryptographic functions
rsa_private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
)
rsa_public_key = rsa_private_key

dsa_private_key = dsa.generate_private_key(key_size=2048)

parameters = dh.generate_parameters(generator=2, key_size=2048)
dh_private_key = parameters.generate_private_key()
dh_public_key = dh_private_key

ec_private_key = ec.generate_private_key(ec.SECP384R1())


algo_dict = {
    # "AES": [algorithms.AES, "key_128", 16],
    # "AES128": [algorithms.AES128, "key_128", 16],
    # "AES256": [algorithms.AES256, "key_256", 16],
    # "DES3": [algorithms.TripleDES, "key_128", 8],
    # "Blowfish": [algorithms.Blowfish, "key_128", 8],
    # "Camellia": [algorithms.Camellia, "key_128", 16],
    # "CAST5": [algorithms.CAST5, "key_128", 8],
    # "ChaCha20": [algorithms.ChaCha20, "key_256", 16],
    # "RC4": [algorithms.ARC4, "key_128", 16],
    # "IDEA": [algorithms.IDEA, "key_128", 8],
    # "SEED": [algorithms.SEED, "key_128", 16],
    "RSA": [rsa_public_key, "key_128"],
    # "DSA": [dsa_private_key, "key_128"],
    # "ECC": [ec_private_key, "key_128"],
}


def encrypt_text(text):
    """Encrypts text using cryptographic functions randomly sampled from the library"""
    padder = PKCS7(128).padder()
    text = padder.update(text.encode('ascii'))
    backend = default_backend()
    key_128 = os.urandom(16)
    key_256 = os.urandom(32)
    nonce = os.urandom(16)
    algo = random.choice(list(algo_dict.keys()))

    if algo_dict[algo][1] == "key_128":
        key = key_128
    else:
        key = key_256
    # print("key: ", key)
    # print("algo: ", algo)
    if algo in ["RSA", "ECC"]:
        pkey = algo_dict[algo][0]
        try:
            ct = pkey.sign(text, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256())), algorithm=hashes.SHA256())
        except Exception as e:
            ct = pkey.sign(text, ec.ECDSA(hashes.SHA256()))

    else:
        try:
            cipher = Cipher(algo_dict[algo][0](key), modes.CBC(b"\x00" * algo_dict[algo][2]), backend=backend)
        except Exception as e:
            try:
                cipher = Cipher(algo_dict[algo][0](key, nonce), None, backend=backend)
            except Exception as e:
                cipher = Cipher(algo_dict[algo][0](key), None, backend=backend)
        encryptor = cipher.encryptor()
        ct = encryptor.update(text) + encryptor.finalize()
    return ct, algo


def read_data_csv(filename):
    """Reads data from a csv file"""
    df = pd.read_csv(filename)
    df.loc[~(df==0).all(axis=1)]
    df = df[df['review/text'].notna()]
    text = df['review/text']
    data = {"encrypted_text": [], "algo": []}
    for i in text:
        ct, algo = encrypt_text(i)
        print(ct)
    #     data['encrypted_text'].append(ct)
    #     data['algo'].append(algo)
    # data = pd.DataFrame(data)
    # data.to_csv("encrypted_data.csv", index=False)

read_data_csv("archive/Books_rating.csv")
# data = pd.read_csv("encrypted_data.csv")
# print(data.head())