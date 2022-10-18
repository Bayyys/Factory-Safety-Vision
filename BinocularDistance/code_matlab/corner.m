%Read the image.

I = imread('img_r11.bmp');
I = rgb2gray(I);
%Find and extract corner features.

corners = detectHarrisFeatures(I);
[features, valid_corners] = extractFeatures(I, corners);
%Display image.

figure; imshow(I); hold on
plot(valid_corners);
