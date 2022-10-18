clc
%读取图像
I1= imread('img_l11.bmp');  
I1=imresize(I1,0.6);  %imresize 将原来的图像缩小原来的一般
I1=rgb2gray(I1);  %把RGB图像变成灰度图像
figure
imshow(I1)
I2= imread('img_r11.bmp');  
I2=imresize(I2,0.6);  
I2=rgb2gray(I2);
figure
imshow(I2)

%寻找特征点  
points1 = detectSURFFeatures(I1);  %读取特征点
points2 = detectSURFFeatures(I2);   

%Extract the features.计算描述向量  
[f1, vpts1] = extractFeatures(I1, points1);  
[f2, vpts2] = extractFeatures(I2, points2);  

%Retrieve the locations of matched points. The SURF feature vectors are already normalized.  
%进行匹配  
indexPairs = matchFeatures(f1, f2, 'Prenormalized', true) ;  
matched_pts1 = vpts1(indexPairs(:, 1)); %这个地方应该是对他进行取值吧 这个应该是啊吧他们做成一个数组
matched_pts2 = vpts2(indexPairs(:, 2));  
%显示匹配
figure('name','匹配后的图像'); showMatchedFeatures(I1,I2,matched_pts1,matched_pts2,'montage');  %总共找了39个特征点
legend('matched points 1','matched points 2'); 
