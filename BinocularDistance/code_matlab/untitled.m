I = imread("Left11.bmp");
thresh = graythresh(I);     %自动确定二值化阈值
J = im2bw(I,thresh);
figure
imshow(J)
