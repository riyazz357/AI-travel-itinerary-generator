import os

# Generate 24 random bytes and convert to hexadecimal
secret_key = os.urandom(24).hex()
print(secret_key)
