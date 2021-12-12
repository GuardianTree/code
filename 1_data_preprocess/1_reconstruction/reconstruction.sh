
#!/bin/sh


# ls *.tgz|xargs -n1 tar zxvf
#recon-all -i ./sample-001.mgz -s study -all

for f in /storage/ZLL/tool/dataV1/1/*
do
    export SUBJECTS_DIR=$f
    recon-all -i $f/baseline/structural/t1_brain.nii.gz -s data -all
done

