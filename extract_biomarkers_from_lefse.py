#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------------------------------------------------
# description
# ---------------------------------------------------------------

"""
lefse.extract_biomarkers_from_lefse.py
============
Please type "./lefse.extract_biomarkers_from_lefse.py -h" for usage help
    
Author:
    Li Hongjun

Description:
    Extraction and accumulation of abundance of biomarkers selected by Lefse.
    Used for the next step.

Reurirements:
    Python packages: argparse,pandas
"""

# ---------------------------------------------------------------
# import
# ---------------------------------------------------------------

import re
import argparse
from pandas import Series
import pandas as pd

# ---------------------------------------------------------------
# function definition
# ---------------------------------------------------------------

def parse_args():
    """ master argument parser """
    
    parser = argparse.ArgumentParser(
        description = "Extraction and accumulation of abundance of biomarkers selected by Lefse.",
        #epilog="",C
        #formatter_class=argparse.RawDescriptionHelpFormatter,
        )
    parser.add_argument(
        '-i', '--input_file',
        type=str,
        required=True,
        help="""
        Lefse LDA res file as input.
        """,
        )
    parser.add_argument(
        '-f', '--abd_file',
        type=str,
        required=True,
        help="""
        Taxa abundance file.
        """,
        )
    parser.add_argument(
        '-o', '--output_file',
        type=str,
        #required=True,
        default="default.output",
        help="""
        Define name of file outputted.
        """,
        )
    
    args = parser.parse_args()
    return args

# ---------------------------------------------------------------
# main function
# -----------------------------------------------------------    ----
def main():
    """ main """
    args = parse_args()
    df = pd.read_table(args.input_file, header = None).dropna(how = 'any')\
    .ix[:,[0,2,3]].sort_values(by=[2,3], ascending=False)
    # Create a pandas object and sort with lda score 
    # dropna: drop rows with NA
    # ascending = F: 降序
    # ix[m:n ,[m,n]] 取m到n行, m和n列
    lables = df[2].unique()
    # Get lable strings
    #print lables
    label_taxa_dict = {}
    for _ele in lables:
        label_taxa_dict[_ele] = [ele.replace(".", "|") for ele in \
                                 df[df[2] == _ele].head(10).ix[:, 0].unique()]
    # label_taxa_dict[label] store the top10 taxa selected by Lefse of group label
    	print len(label_taxa_dict[_ele])
    df = pd.read_table(args.abd_file, header = None)
    # Abundance table
    for _group in label_taxa_dict.keys():
        df_group = pd.DataFrame()
        df_group = df_group.append(df.ix[0])
        for _taxa in label_taxa_dict[_group]:
            _taxa_abd_series = df[df[0].str.find(_taxa) == 0].ix[:,1:]\
            .astype(float).sum()
            # 包含marker物种的每个样本的丰度加和并写入DF一行
            #### 此处需根据实际情况进行更改 ####
            _taxa_species = re.split(r"\|", _taxa)[-1]
            ##################################
            temp_series = Series([_taxa_species]).append(_taxa_abd_series)
            df_group = df_group.append(temp_series, ignore_index = True)
            # 把 group组的marker丰度加和并加入到总的df
        df_group.T.to_csv("%s_biomarker.csv" % _group, header = False, index = False)
    
if __name__ == "__main__":
    main()
