#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Richard

import argparse
import re
import random

def parse_args():
    """ master argument parser """
    parser = argparse.ArgumentParser(
        description = """        
        This script is used for dividing file to train and test datasets followed by 2-8 rule.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        )
    parser.add_argument(
        '-f', '--file_name',
        type=str,
        required=True,
        help="""
        Input file name to be divided.
        """,
    )
    parser.add_argument(
        '-g', '--group_name',
        type=str,
        required=True,
        help="""
        Input group name which mei may be bigger relatively.
        """,
    )
    parser.add_argument(
        '-a', '--another_group_name',
        type=str,
        required=True,
        help="""
        Input another group name.
        """,
    )
    args = parser.parse_args()
    return args

def rand_div(list, leng):
    test_len = int(0.2 * leng)
    while test_len > 0:
        ran = random.randint(0, leng-1)
        tmp = list[ran]
        list[ran] = list[-1]
        list[-1] = tmp
        list.pop()
        leng = leng - 1
        test_len = test_len -1
        with open("test.txt", 'a') as test:
            test.write(tmp + "\n")
    
    str = "\n".join(list)
    with open("train.txt", 'a') as train:
        train.write(str + "\n")


def divide(args):
    f1_list = []
    f2_list = []
    with open(args.file_name, 'r') as file_input:
        first_line = file_input.readline()
        first_line = re.sub(r"_\S+", "", first_line.strip())
        with open("test.txt", 'a') as test:
            test.write(first_line + "\n")
        with open("train.txt", 'a') as train:
            train.write(first_line + "\n")

        for line in file_input.readlines():
            if line.startswith(args.group_name):
                line = re.sub(r"^\S+\t", r"1\t", line.strip(), count=1)
                f1_list.append(line)
            elif line.startswith(args.another_group_name):
                line = re.sub(r"^\S+\t", r"-1\t", line.strip(), count=1)
                f2_list.append(line)
            else:
                print "Wrong, more groups than input!"
    f1_len = len(f1_list)
    f2_len = len(f2_list)
    rand_div(f1_list, f1_len)
    rand_div(f2_list, f2_len)


        


def main():
    args = parse_args()
    divide(args)
    


if __name__ == "__main__":
    main()
