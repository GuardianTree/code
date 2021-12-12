%% add path
addpath('/storage/ZLL/3-hinge/tmp/lib/gyralnet/freesurfer_matlab');
addpath('/storage/ZLL/3-hinge/tmp/lib/gyralnet/gyriCrest');
addpath('/storage/ZLL/3-hinge/tmp/lib/Surface');
addpath('/storage/ZLL/3-hinge/tmp/lib/gyralnet/myVtk');
addpath('/storage/ZLL/3-hinge/tmp/lib/own');
%% set data dir
%data_dir='/storage/caoxiaoling/tool/freesufer/datadir/';
%F='/storage/caoxiaoling/tool/freesufer/datadir/';
%F='/storage/caoxiaoling/test/done/';
F='/storage/student22/code/3hinge_test.xlsx';
Out='/storage/student22/code/see/';

%[hinge2_vtx,Txt,Raw]=xlsread(F)
hinge2=coorltestA(:,1:3)
hinge2_vtx=hinge2'
sphere2=sphere_group(hinge2_vtx,1);
SurfWrite([Out 'face_l.vtk'],sphere2,[]);
save([Out 'face_l.mat'],'hinge2*');
