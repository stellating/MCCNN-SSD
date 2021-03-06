# -*- coding: utf-8 -*-  
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES, PKCS1_OAEP
import os
import base64

def Encrypt(path,name): 
    filename = os.path.join(path,name)         
    data = ''
    with open(filename, 'rb') as f:
        data = f.read()
    with open(filename, 'wb') as out_file:
        # 收件人秘钥 - 公钥
        recipient_key = RSA.import_key(open(os.path.join(path,'my_rsa_public.pem')).read())
        session_key = get_random_bytes(16)
        # Encrypt the session key with the public RSA key
        cipher_rsa = PKCS1_OAEP.new(recipient_key)
        out_file.write(cipher_rsa.encrypt(session_key))
        # Encrypt the data with the AES session key
        cipher_aes = AES.new(session_key, AES.MODE_EAX)
        ciphertext, tag = cipher_aes.encrypt_and_digest(data)
        out_file.write(cipher_aes.nonce)
        out_file.write(tag)
        out_file.write(ciphertext)
        
def Descrypt(path,name):
    filename = os.path.join(path,name) 
    code = 'nooneknows'
    with open(filename, 'rb') as fobj:
        private_key = RSA.import_key(open(os.path.join(path,'my_private_rsa_key.bin')).read(), passphrase=code)
        enc_session_key, nonce, tag, ciphertext = [ fobj.read(x) 
                                                    for x in (private_key.size_in_bytes(), 
                                                    16, 16, -1) ]
        cipher_rsa = PKCS1_OAEP.new(private_key)
        session_key = cipher_rsa.decrypt(enc_session_key)
        cipher_aes = AES.new(session_key, AES.MODE_EAX, nonce)
        data = cipher_aes.decrypt_and_verify(ciphertext, tag)
    
    with open(filename, 'wb') as wobj:
        wobj.write(data) 
def RenameFile(dir,filename):
    filename_bytes = filename.encode('utf-8')
    filename_bytes_base64 = base64.encodestring(filename_bytes)
    
    filename_bytes_base64 = filename_bytes_base64[::-1][1:]
    new_filename = filename_bytes_base64.decode('utf-8') + '.crypt1'
    print(os.path.join(dir, filename))
    print(os.path.join(dir,new_filename))
    os.rename(os.path.join(dir, filename), os.path.join(dir,new_filename))
def ReserveFilename(dir, filename):
    f = filename
    filename = filename[::-1][7:][::-1]
    filename_base64 = filename[::-1] + '\n'
    filename_bytes_base64 = filename_base64.encode('utf-8')
    ori_filename = base64.decodestring(filename_bytes_base64).decode('utf-8')
    print(os.path.join(dir, f))
    print(os.path.join(dir,ori_filename))
    os.rename(os.path.join(dir, f),os.path.join(dir,ori_filename))
    



