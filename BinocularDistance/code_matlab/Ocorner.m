I = imread('img_l11.bmp');
I = rgb2gray(I);

%Find and extract features.
points = detectSURFFeatures(I);%I是输入的灰度图像，返回值是一个 SURFPoints类，这个SURFPoints类包含了一些从这个灰度图像中提取的一些特征
[features, valid_points] = extractFeatures(I, points);

%Display and plot ten strongest SURF features.
figure; imshow(I); hold on;
plot(valid_points.selectStrongest(10),'showOrientation',true);
