function [img_warped, hrnew, gvnew] =  normalizeImg(inputImg, target_3D, hR, gc, roiSize, cameraMatrix, focal_new, distance_new)
     % prepare the data
     % for right eye
     if nargin < 8
        focal_new=960; % facoal length of the virual camera can be changed as needed.
     end
     if nargin < 9
        distance_new=600; % please do not change it or set it to be different value, otherwise the gaze label will be different.
     end
     distance = norm(target_3D);
     z_scale = distance_new/distance;
     cam_new = [focal_new, 0, roiSize(1)/2; 0.0, focal_new, roiSize(2)/2; 0, 0, 1.0];
     scaleMat = [1.0, 0.0, 0.0; 0.0, 1.0, 0.0; 0.0, 0.0, z_scale];
     hRx = hR(:,1);
     forward = (target_3D/distance);
     down = cross(forward, hRx);
     down = down / norm(down);
     right = cross(down, forward);
     right = right / norm(right);
     rotMat = [right, down, forward]';
     warpMat = (cam_new* scaleMat) * (rotMat * inv(cameraMatrix) );
     img_warped = cv.warpPerspective(inputImg, warpMat, 'DSize', roiSize);     
     
     % rotatoin normalization
     cnvMat = scaleMat * rotMat;
     hRnew = cnvMat * hR;
     hrnew = rodrigues(hRnew);
     htnew = cnvMat * target_3D;
     
     % gaze vector normalizatoin
     gcnew = cnvMat * gc;
     gvnew = gcnew -  htnew;
     gvnew = gvnew / norm(gvnew);
     
end