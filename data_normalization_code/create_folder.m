clear  ;
clc  ;

%cd('F:\datebase\zhang xucong_paper\MPIIGaze\MPIIGaze\Data\Original_crop\p00');%���õ�ǰĿ¼��current directory   

%for i = 0:14
    
    folderName=strcat( 'F:\datebase\zhang_xucong_paper\MPIIGaze\MPIIGaze\Data\gaze_2D\P',int2str(i),'');
  i=14
    for j = 1:7
        
        folderName_l=strcat( 'F:\datebase\zhang_xucong_paper\MPIIGaze\MPIIGaze\Data\gaze_2D\P',int2str(i),'\day',int2str(j),'\left');
        folderName_r=strcat( 'F:\datebase\zhang_xucong_paper\MPIIGaze\MPIIGaze\Data\gaze_2D\P',int2str(i),'\day',int2str(j),'\right');
        mkdir(folderName_l);  % �½�һ���ļ���   
        mkdir(folderName_r);

    end  
