for f in /storage/ZLL/code/FSL/Flirt/*
do
    echo "$f"
    flirt -in /storage/ZLL/code/FSL/template/data876/t1_brain.nii.gz -ref $f/t1_brain.nii.gz -out /storage/ZLL/code/FSL/registrated_trash.nii.gz -omat /storage/ZLL/code/FSL/876.mat -dof 6 -interp nearestneighbour
    flirt -in /storage/ZLL/code/FSL/template/data876/surf0_l.nii.gz -ref $f/t1_brain.nii.gz -out $f/B_label_l.nii.gz -init /storage/ZLL/code/FSL/876.mat -applyxfm -interp nearestneighbour

done


