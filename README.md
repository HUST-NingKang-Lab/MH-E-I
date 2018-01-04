#MH(E)I
Scripts for MH(E)I (Microbial-based Human / Environment Index) classifer.

##Requirement
*Python
*Python libraries: pandas, matplotlib, sklearn, scipy, argparse, random, time, re
*R
*R libraries: argparse, pROC

##Console command line:
###run lefse to select legacy biomarkers.
>format_input.py input_abundance.file lefse.formatted.in -c 1 -s -1 -u 2 -o 1000000
>run_lefse.py lefse.formatted.in lefse.lda.res -y 1 #-l $2
>plot_res.py lefse.lda.res bar.plot.svg --format svg --dpi 300

###run this script to extract and accumulate the abundance of legacy biomarkers selected by Lefse.
>python extract_biomarkers_from_lefse.py -i lefse.lda.res -f $1

###run mRMR to select reprentative biomarkers from legacy biomarkers.
>mrmr -i ${3}_biomarker.csv -m MID -n 5 -t 1 | tee ${3}_biomarker_mrmr.txt
>mrmr -i ${4}_biomarker.csv -m MID -n 5 -t 1 | tee ${4}_biomarker_mrmr.txt

###run this script to extract the abundance of legacy biomarkers selected by mRMR.
>python extract_features_from_mrmr.py -i ${3}_biomarker_mrmr.txt ${4}_biomarker_mrmr.txt -t ${3}_biomarker.csv ${4}_biomarker.csv 

###run this command to remove txts if they exist.
>rm test.txt train.txt

###run this script to divide samples to train(80%) and test(20%) datasets.
>python divide28.py -f mrmr_features_abd.txt -g $5 -a $6

###This script is used for train and test mei model.
###MH(E)I model: MH(E)I score = (∑(i=1-n){ABU_r}(Si))/(∑(j=1-n){ABU_r}(Sj))
>python me_h_i_model.py -f train.txt -t test.txt -g $5

#run this this script to plot ROC curve for MH(E)I
>Rscript plot_roc.R -m test_mei_list_for_roc_r.txt -f mrmr_features_abd.txt
