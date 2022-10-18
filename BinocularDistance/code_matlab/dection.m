%step1:读取图片
%读取object图片
boxImage = rgb2gray(imread('img_l10.bmp'));
%读取场景图片
sceneImage = rgb2gray(imread('img_r10.bmp'));

%step2：检测特征点
boxPoints = detectSURFFeatures(boxImage);
scenePoints = detectSURFFeatures(sceneImage);

% figure; imshow(boxImage);
% title('Box Image中最强的100个feature points');
% hold on;
% plot(boxPoints.selectStrongest(100));

%step3 extract feature descriptors  提取出特征的描述子
%使用extractFeatures()，具体的feature类型是通过boxPoints位置的参数指定的，这里是SURF
%烂设计，为什么extractFeatures输入了boxPoints后，还要返回boxPoints?
[boxFeatures, boxPoints] = extractFeatures(boxImage, boxPoints);
[sceneFeatures, scenePoints] = extractFeatures(sceneImage, scenePoints);

%step4 find putative point matches
%Match the features using their descriptors.
boxPairs = matchFeatures(boxFeatures, sceneFeatures);
%Display putatively matched features.
matchedBoxPoints = boxPoints(boxPairs(:,1), :);
matchedScenePoints = scenePoints(boxPairs(:,2),:);
figure;
showMatchedFeatures(boxImage, sceneImage, matchedBoxPoints, matchedScenePoints, 'montage');
title('Putatively Matched Points (Including Outliers)');

%step5 locate the Object in the Scene Using Putative Matches
[tform, inlierBoxPoints, inlierScenePoints] = ...
    estimateGeometricTransform(matchedBoxPoints, matchedScenePoints, 'affine');
figure;
showMatchedFeatures(boxImage, sceneImage, inlierBoxPoints, ...
    inlierScenePoints, 'montage');
title('Matched Points (Inliers Only)');

%Get the bounding polygon of the reference image.
boxPolygon = [1, 1;... % top-left
    size(boxImage,2), 1; ... % top-right
    size(boxImage, 2), size(boxImage, 1); ... % bottom-right
    1, size(boxImage, 1); ... % bottom-left
    1, 1]; % top-left again to close the polygon

% transform the polygon into the coordinate system of the target image
%将多边形变换到目标图片上，变换的结果表示了物体的位置
newBoxPolygon = transformPointsForward(tform, boxPolygon);

%display the detected object 显示被检测到的物体
figure; imshow(sceneImage);
hold on;
line(newBoxPolygon(:, 1), newBoxPolygon(:, 2), 'Color', 'y');
title('Detected Box');
