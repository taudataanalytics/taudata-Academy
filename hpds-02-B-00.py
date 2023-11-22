# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 12:59:23 2020
Ilustrasi Pengaruh GIL (single thread)
@author: Taufik Sutanto
"""
# single_threaded.py
import time

COUNT = 50000000

def countdown(n):
    while n>0:
        n -= 1

start = time.time()
countdown(COUNT)
end = time.time()

print('Time taken in seconds -', end - start)