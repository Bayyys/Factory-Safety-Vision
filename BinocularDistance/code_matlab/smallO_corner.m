I = imread('img_r11.bmp');
I = rgb2gray(I);

%Find features using MSER with SURF feature descriptor.
regions = detectMSERFeatures(I);
[features, valid_points] = extractFeatures(I,regions,'Upright',true);
%Display SURF features corresponding to the MSER ellipse centers.
figure; imshow(I); hold on;
plot(valid_points,'showOrientation',true);
