# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 08:59:27 2020
Contoh Threading sederhana di Python 3
@author: Taufik Sutanto
"""
import time
from threading import Thread

def sleeper(i):
    nSleep = 3
    print("thread {} sleeps for {} seconds".format(i, nSleep))
    time.sleep(nSleep)
    print("thread %d woke up" % i)

if __name__ == "__main__":
    for i in range(10):
        t = Thread(target=sleeper, args=(i,))
        t.start()