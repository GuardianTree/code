%% add path
addpath('/storage/ZLL/3-hinge/tmp/lib/gyralnet/freesurfer_matlab');
addpath('/storage/ZLL/3-hinge/tmp/lib/gyralnet/gyriCrest');
addpath('/storage/ZLL/3-hinge/tmp/lib/Surface');
addpath('/storage/ZLL/3-hinge/tmp/lib/gyralnet/myVtk');
addpath('/storage/ZLL/3-hinge/tmp/lib/own');
%% set data dir
F='/storage/student22/code/3/data';
path1='/storage/student22/hinge/3hinge/lh/3h_num';
path2='/storage/student22/hinge/2hinge/lh/2h_num';
path3='/storage/student22/hinge/3hinge/lh/3h_';
path4='/storage/student22/hinge/2hinge/lh/2h_';


a=0;
p=0;
C=[];
C1=[];
D=[];
D1=[];
tt ='/ww';
h=1;

for kk=0:5
    A=[];
    A2=[];
    h1=1;
    for i=1000+a:1019+a
        data_di=[F,num2str(i)];
        data_dir=strcat(data_di,tt)
        fname_white=[data_di '/lh.smoothwm'];
        
        fname_sulc=[data_di '/lh.sulc'];
        
        fname_thickness=[data_di '/lh.thickness'];
        
       
        
        parameters=[];
        mkdir([data_dir '/lh']);
        fname_out=[data_dir '/lh'];
        [gyriNetwork,surf]=coExtractGyriNetwork(fname_white, fname_sulc, fname_thickness, parameters, fname_out);
        %save([fname_out '_gyralnetwork.mat'],'gyriNetwork');
        %save([fname_out '_surf.mat'],'surf');
        %load([fname_out '_surf.mat']);
        %load([fname_out '_gyralnetwork.mat']);
        svertice = surf.Vtx;
        wvertice = svertice';
        wroi=surf.roi;
        wgyrus = surf.gyrus;
        wsurf.vertice =  surf.Vtx;
        wsurf.faces   =  surf.Face-1;
        wsurf.POINT_DATA.SCALARS=[];
        wsurf.POINT_DATA.VECTORS=[];
        wsurf.CELL_DATA.SCALARS=[];
        wsurf.CELL_DATA.VECTORS=[];
        SurfWrite([fname_out '_surf.vtk'],wsurf,[]);


        [x,y]=find(gyriNetwork~=0);
        uy=unique(y);
        hinge_num=zeros(length(uy),1);
        for j=1:length(uy)
            hinge_num(j)=length(find(y==uy(j)));
        end
        %%fprintf('hinge_num is %d\n',vertices);
        
        hinge3_idx=uy(hinge_num==3);
        vertices_3hinge=wsurf.vertice(:,hinge3_idx);
        hinge3_vtx = vertices_3hinge';
        data=[hinge3_idx,hinge3_vtx];
        [m,n]=size(data);
        data_cell=mat2cell(data,ones(m,1),ones(n,1));
        title={'hinge3_idx','hinge3_vtx1','hinge3_vtx2','hinge3_vtx3'};
        result=[title;data_cell];
        [x1,x2]=size(result);
        [y1,y2]=size(A);
        if y1~=0
            if (x1>y1)
                c=x1-y1;
                B1=zeros(c,y2);
                B=num2cell(B1);
                A=[A;B];
            else
                c=y1-x1;
                B1=zeros(c,x2);
                B=num2cell(B1);
                [c1,c2]=size(B);
                [v1,v2]=size(result);
                result=[result;B];
            end
        end
        A=cat(2,A,result);
        
        hinge2_idx=uy(hinge_num==2);
        vertices_2hinge=wsurf.vertice(:,hinge2_idx);
        hinge2_vtx = vertices_2hinge';
        data2=[hinge2_idx,hinge2_vtx];
        [m2,n2]=size(data2);
        data_cell2=mat2cell(data2,ones(m2,1),ones(n2,1));
        title2={'hinge2_idx','hinge2_vtx1','hinge2_vtx2','hinge2_vtx3'};
        result2=[title2;data_cell2];
        [x12,x22]=size(result2);
        [y12,y22]=size(A2);
        if y12~=0
            if (x12>y12)
                c2=x12-y12;
                B12=zeros(c2,y22);
                B2=num2cell(B12);
                A2=[A2;B2];
            else
                c2=y12-x12;
                B12=zeros(c2,x22);
                B2=num2cell(B12);
                [c12,c22]=size(B2);
                [v12,v22]=size(result2);
                result2=[result2;B2];
            end
        end
        A2=cat(2,A2,result2);
        
        
        
        sphere=sphere_group(vertices_3hinge,1);
        SurfWrite([fname_out '_3hinge.vtk'],sphere,[]);
        
        C1(h1)=length(hinge3_idx);
        D1(h1)=length(hinge2_idx);
        
        h1=h1+1;
        
        %%save([fname_out '_3hinge.mat'],'hinge3*');
        %%save([fname_out '_numhinge.mat'],'hinge_num');
    end
   
    path1_1 = [path1,num2str(kk)];
    txt_3hinge =[path1_1, '.txt'];
    fid = fopen(txt_3hinge,'wt');
    fprintf(fid,'%d\t',C1);
    fclose(fid);
    
    path2_2 = [path2,num2str(kk)];
    txt_2hinge =[path2_2, '.txt'];
    fid2 = fopen(txt_2hinge,'wt');
    fprintf(fid2,'%d\t',D1);
    fclose(fid2);
    
    [a1,a2]=size(A)
    path3_3 = [path3,num2str(kk)];
    txtname =[path3_3, '.txt'];
    pfile = fopen(txtname,'wt');
    ee = 0;
    while pfile ==-1
        pfile = fopen(txtname,'wt');
        ee=ee+1
    end
    %fclose(pfile);
    for p= 2:a1
        %fprintf(pfile,'%s',A{1,l});
        for l =1:4:a2
            %save(pfile,'A{p,l}');
            kl0 = l;
            fprintf(pfile,'%d\t',A{p,l});
            kl1=l+1;
            fprintf(pfile,'%.4f\t',A{p,kl1});
            kl2=l+2;
            fprintf(pfile,'%.4f\t',A{p,kl2});
            kl3=l+3;
            fprintf(pfile,'%.4f\t',A{p,kl3});
            %%kl4=l+4;
            %%fprintf(pfile,'%.4f\t',A{p,kl4});
        end
        fprintf(pfile,'\r\n');
    end
    st=fclose(pfile);
    gg=0;
    while st==-1
        st = fclose(pfile);
        gg=gg+1
    end

    
    [a12,a22]=size(A2)
    path4_4 = [path4,num2str(kk)];
    txtname2 =[path4_4, '.txt'];
    pfile2 = fopen(txtname2,'wt');
    ee2=0;
    while pfile2 ==-1
        pfile2 = fopen(txtname2,'wt');
        ee2= ee2+1
    end
    %fclose(pfile);
    for p2= 2:a12
        %fprintf(pfile,'%s',A{1,l});
        for l2 =1:4:a22
            
            kl02 = l2;
            fprintf(pfile2,'%d\t',A2{p2,l2});
            kl12=l2+1;
            fprintf(pfile2,'%.4f\t',A2{p2,kl12});
            kl22=l2+2;
            fprintf(pfile2,'%.4f\t',A2{p2,kl22});
            kl32=l2+3;
            fprintf(pfile2,'%.4f\t',A2{p2,kl32});
            %%kl4=l+4;
            %%fprintf(pfile,'%.4f\t',A{p,kl4});
        end
        fprintf(pfile2,'\r\n');
    end    
    st2 =fclose(pfile2);
    gg2 = 0;
    while st2==-1
        st2 = fclose(pfile2);
        gg2 = gg2+1
    end

 
    a=a+20;
end
