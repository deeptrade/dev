'''
Download s&p500 stock data, using symbols described in the input file.
The input file should be a csv with the first column being the stock symbols. 
'''

import argparse
import urllib
import os

BASEURL="https://www.quandl.com/api/v3/datatables/WIKI/PRICES.json?qopts.columns=ticker,date,adj_close,adj_volume"

parser = argparse.ArgumentParser(description='downloading data from quandl')
parser.add_argument("-i", "--input", help="Set input file name.")
parser.add_argument("-o", "--output", help="Set output directory name.")
parser.add_argument("-k", "--key", help="Set the quandl api key")

args = parser.parse_args()

if args.output == None or args.input == None or args.key == None:
    print("argements not specified, run with -h to see the help")
    exit()

with open(args.input, 'r') as fin:
    fin.readline() # skip the first line, which is the header
    for line in fin.readlines():
        ar = line.split(',')

        print("reading {}\n".format(ar[0]))
        urlstr = "{}&api_key={}&ticker={}".format(BASEURL, args.key, ar[0])
        urllib.urlretrieve(urlstr, "{}/{}.json".format(args.output, ar[0]))

