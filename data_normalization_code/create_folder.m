clear  ;
clc  ;

%cd('F:\datebase\zhang xucong_paper\MPIIGaze\MPIIGaze\Data\Original_crop\p00');%设置当前目录：current directory   

%for i = 0:14
    
    folderName=strcat( 'F:\datebase\zhang_xucong_paper\MPIIGaze\MPIIGaze\Data\gaze_2D\P',int2str(i),'');
  i=14
    for j = 1:7
        
        folderName_l=strcat( 'F:\datebase\zhang_xucong_paper\MPIIGaze\MPIIGaze\Data\gaze_2D\P',int2str(i),'\day',int2str(j),'\left');
        folderName_r=strcat( 'F:\datebase\zhang_xucong_paper\MPIIGaze\MPIIGaze\Data\gaze_2D\P',int2str(i),'\day',int2str(j),'\right');
        mkdir(folderName_l);  % 新建一个文件夹   
        mkdir(folderName_r);

    end  
