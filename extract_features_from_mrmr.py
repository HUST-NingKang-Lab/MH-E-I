#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------------------------------------------------
# description
# ---------------------------------------------------------------

"""
extract_features_from_mrmr.py
============
Please type "./extract_features_from_mrmr.py -h" for usage help
    
Author:
    Li Hongjun

Description:
    context

Reurirements:
    Python packages: argparse, re
"""

# ---------------------------------------------------------------
# import
# ---------------------------------------------------------------

import re
import argparse
import pandas as pd

# ---------------------------------------------------------------
# function definition
# ---------------------------------------------------------------

def parse_args():
    """ master argument parser """
    parser = argparse.ArgumentParser(
        description = "",
        #epilog="",
        #formatter_class=argparse.RawDescriptionHelpFormatter,
        )
    parser.add_argument(
        '-i', '--mrmr_result',
        type=str,
        required=True,
        nargs="+",
        help="""
        Transmit file inputted to the script.
        """,
        )
    parser.add_argument(
        '-t', '--abd_table',
        type=str,
        required=True,
        nargs="+",
        help="""
        Abundance Table of biomarker(csv).
        """,
        )
    
    args = parser.parse_args()
    return args

# ---------------------------------------------------------------
# main function
# ---------------------------------------------------------------

def main():
    """ main """
    args = parse_args()
    for i in range(2):
        with open(args.mrmr_result[i]) as mrmr_result:
        # 获取mrmr方法得到的特征
            file_cont_str = mrmr_result.read()
            mrmr_list = re.search(r"\*\*\* mRMR features \*\*\*([\s\S]+)\*\*\*",\
                                  file_cont_str).group(1).split("\n")[2:-3]
            feature_names_list = [ele.split("\t")[2][1:-1] for ele in mrmr_list]
        
            df = pd.read_csv(args.abd_table[i])
            # 特征写入文件
            if i == 0:
                df_temp = pd.DataFrame(df.ix[:,0])
            else:
                pass
            group_name = args.mrmr_result[i].split("_")[0]
            for ele in feature_names_list:
                df_temp["%s_%s" % (group_name, ele)] = df.ix[:, ele]
    
    df_temp.to_csv("mrmr_features_abd.txt", sep = "\t" ,index=False)
    
    
if __name__ == "__main__":
    main()