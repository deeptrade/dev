#!/usr/local/bin/python3

import os
import subprocess

docker_proc = subprocess.Popen(["docker ps -a | grep month"], stdout=subprocess.PIPE, shell=True)
(out, err) = docker_proc.communicate()

lines = out.split(b'\n')
for line in lines:
    ar = line.decode("utf-8").split()
    if len(ar) > 0:
        os.system("docker rm " + ar[0])

docker_proc = subprocess.Popen(["docker images | grep month"], stdout=subprocess.PIPE, shell=True)
(out, err) = docker_proc.communicate()

lines = out.split(b'\n')
for line in lines:
    ar = line.decode("utf-8").split()
    if len(ar) > 2:
        os.system("docker rmi " + ar[2])

