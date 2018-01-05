#sh run.sh icu.txt A B A B ../
#lefse
format_input.py ${1} lefse.formatted.in -c 1 -s -1 -u 2 -o 1000000
run_lefse.py lefse.formatted.in lefse.lda.res -y 1
#plot_res.py lefse.lda.res bar.plot.svg --format svg --dpi 300

python ${7}extract_biomarkers_from_lefse.py -i lefse.lda.res -f $1
#mRMR
mrmr -i ${2}_biomarker.csv -m MID -n 5 -t 1 | tee ${2}_biomarker_mrmr.txt
mrmr -i ${3}_biomarker.csv -m MID -n 5 -t 1 | tee ${3}_biomarker_mrmr.txt
python ${6}extract_features_from_mrmr.py -i ${2}_biomarker_mrmr.txt ${3}_biomarker_mrmr.txt -t ${2}_biomarker.csv ${3}_biomarker.csv 
#mei
rm test.txt train.txt
python ${6}divide28.py -f mrmr_features_abd.txt -g $4 -a $5
python ${6}me_h_i_model.py -f train.txt -t test.txt -g $4
#plot
Rscript ${6}plot_roc.R -m test_mei_list_for_roc_r.txt -f mrmr_features_abd.txt
