clear;
clc;

addpath('F:/mexopencv-3.4/mexopencv-master/mexopencv-master/'); % we need to use some function from OpenCV

% load the face model
faceModel = load('F:/datebase/zhang_xucong_paper/MPIIGaze/MPIIGaze/6 points-based face model.mat');
faceModel = faceModel.model;

% load the image, annotation and camera parameters.
% img = imread('F:/datebase/zhang xucong_paper/MPIIGaze/MPIIGaze/Data/Original/p00/day01/0001.jpg');
% annotation = load('F:/datebase/zhang xucong_paper/MPIIGaze/MPIIGaze/Data/Original/p00/day01/annotation.txt');
% cameraCalib = load('F:/datebase/zhang xucong_paper/MPIIGaze/MPIIGaze/Data/Original/p00/Calibration/Camera.mat');


file_path_person =  'F:\datebase\zhang xucong_paper\MPIIGaze\MPIIGaze\Data\Original\';
%person_path_list = dir(file_path_person);
%person_num = length(person_path_list);

%for k = 1:(person_num-3)
    k = 0
    file_path_day =  strcat('F:\datebase\zhang_xucong_paper\MPIIGaze\MPIIGaze\Data\Original\p',int2str(k),'\');
    day_path_list = dir(file_path_day);
    day_num = length(day_path_list);
    for i = 1:(day_num -3)

        file_path = strcat('F:/datebase/zhang_xucong_paper/MPIIGaze/MPIIGaze/Data/Original/p',int2str(k),'/day',int2str(i),'/');
        annotation_path = strcat('F:/datebase/zhang_xucong_paper/MPIIGaze/MPIIGaze/Data/Original/p',int2str(k),'/day',int2str(i),'/annotation.txt');
        cameraCalib_path = strcat('F:/datebase/zhang_xucong_paper/MPIIGaze/MPIIGaze/Data/Original/p',int2str(k),'/Calibration/Camera.mat');

        annotation = load(annotation_path)
        cameraCalib = load(cameraCalib_path)
        img_path_list = dir(strcat(file_path,'*.jpg'));
        img_num = length(img_path_list);
        if img_num > 0 
            for j = 1:img_num
                image_name = img_path_list(j).name;
                img =  imread(strcat(file_path,image_name));  
        % get head pose
                headpose_hr = annotation(j, 30:32);  %Modified parameters
                headpose_ht = annotation(j, 33:35);   %Modified parameters
                hR = rodrigues(headpose_hr); 
                Fc= hR* faceModel; % rotate the face model, which is calcluated from facial landmakr detection
                Fc= bsxfun(@plus, Fc, headpose_ht');  %Fc size 3*6

        % get the eye center in the original camera cooridnate system.
                right_eye_center = 0.5*(Fc(:,1)+Fc(:,2));
                left_eye_center = 0.5*(Fc(:,3)+Fc(:,4));

        % get the gaze target
                gaze_target = annotation(j, 27:29);
                gaze_target = gaze_target';

        % set the size of normalized eye image
                eye_image_width  = 60;
                eye_image_height = 36;

        % normalization for the right eye, you can do it for left eye by replacing
        % "right_eye_cetter" to "left_eye_center"
                [eye_img_R, headpose_R, gaze_R] = normalizeImg(img, right_eye_center, hR, gaze_target, [eye_image_width, eye_image_height], cameraCalib.cameraMatrix);
           % imshow(eye_img);
                path=strcat('F:\datebase\zhang_xucong_paper\MPIIGaze\MPIIGaze\Data\Original_eye\p',int2str(k),'\day',int2str(i),'\right\');
                pathfile=fullfile(path,image_name);
                imwrite(eye_img_R,pathfile,'jpg');
                fprintf('%d %s\n',j,strcat(file_path,image_name));


                 [eye_img_L, headpose_L, gaze_L] = normalizeImg(img, left_eye_center, hR, gaze_target, [eye_image_width, eye_image_height], cameraCalib.cameraMatrix);
           % imshow(eye_img);
                 path=strcat('F:\datebase\zhang_xucong_paper\MPIIGaze\MPIIGaze\Data\Original_eye\p',int2str(k),'\day',int2str(i),'\left\');
                 pathfile=fullfile(path,image_name);
                 imwrite(eye_img_L,pathfile,'jpg');
                 fprintf('%d %s\n',j,strcat(file_path,image_name));
% convert the gaze direction in the camera cooridnate system to the angle
% in the polar coordinate system
            end
        end
 end
%end
% convert the gaze direction in the camera cooridnate system to the angle
% in the polar coordinate system
%gaze_theta = asin((-1)*gaze(2)); % vertical gaze angle
%gaze_phi = atan2((-1)*gaze(1), (-1)*gaze(3)); % horizontal gaze angle

% save as above, conver head pose to the polar coordinate system
%M = rodrigues(headpose);
%Zv = M(:,3);
%headpose_theta = asin(Zv(2)); % vertical head pose angle
%headpose_phi = atan2(Zv(1), Zv(3)); % horizontal head pose angle
