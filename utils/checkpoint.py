
'''
Check whether they are any new checkpoints, and save them to S3
'''

import os
import argparse
import time

parser = argparse.ArgumentParser(description='downloading data from quandl')
parser.add_argument("-i", "--input", help="Set the directory that we want to check")
parser.add_argument("-n", "--name", help="The name we give to this series of checkpoints")
parser.add_argument("-b", "--bucket", help="The s3 bucket to save to")

args = parser.parse_args()
if args.input == None:
    print("Input directory must be specified. Please run with -h to see the options.")
    exit(1)
if args.name == None:
    print("You need to give the checkpoint a name. Please run with -h to see the options.")
    exit(1)
if args.bucket == None:
    print("You need provide the S3 bucket name. Please run with -h to see the options.")
    exit(1)

# Do the simple thing for now, checkpoint once every 10 minutes
while True:
    status = os.system("tar cvzf /tmp/{}.tgz {}".format(args.name, args.input))
    if status != 0:
        print("tar command failed")
        exit(status)

    status = os.system("aws s3 cp /tmp/{}.tgz s3://{}".format(args.name, args.bucket))
    if status != 0:
        print("aws s3 cp command failed")
        exit(status)

    time.sleep(600)
