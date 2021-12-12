%% add path
addpath('/storage/ZLL/3-hinge/tmp/lib/gyralnet/freesurfer_matlab');
addpath('/storage/ZLL/3-hinge/tmp/lib/gyralnet/gyriCrest');
addpath('/storage/ZLL/3-hinge/tmp/lib/Surface');
addpath('/storage/ZLL/3-hinge/tmp/lib/gyralnet/myVtk');
addpath('/storage/ZLL/3-hinge/tmp/lib/own');
path1='/storage/ZLL/tool/test/3/data';

%path2='/storage/student22/code';
txtname='/storage/ZLL/tool/test/feature/sulc/sulc';
a=0;
name_title='data';
for j = 1000
    A=[];
    for i=0
        % set file path
        data_dir=[path1,num2str(i)];
        fprintf('data_dir is %s\n',data_dir);
        fname_sulc = [data_dir '/rh.sulc'];
        fname_out=[data_dir '/gyralnet/rh'];
        % i don't know
        asulc = read_curv(fname_sulc);
        [vertices,label,colortable]=read_annotation([data_dir '/rh.aparc.annot']);
        data=[asulc];
        [m,n] = size(data);
        data_cell=mat2cell(data,ones(m,1),ones(n,1));
        area_name=[name_title,num2str(i)];
        title={area_name};
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
    end
    a=a+1000;

    [a1,a2]=size(A)
    path3=[txtname,num2str(j)]
    path4 =[path3, '.txt'];
    pfile = fopen(path4,'wt');
    %fclose(pfile);
    for p= 1
        %fprintf(pfile,'%s',A{1,l});
        for l =1:a2
            %save(pfile,'A{p,l}');
            kl0 = l;
            fprintf(pfile,'%s\t',A{p,l});
            
        end
        fprintf(pfile,"\r\n");
    end
    for p= 2:a1
        %fprintf(pfile,'%s',A{1,l});
        for l =1:a2
            %save(pfile,'A{p,l}');
            kl0 = l;
            fprintf(pfile,'%.4f\t',A{p,l});
        end
        fprintf(pfile,"\r\n");
    end
    
    fclose(pfile);
end

