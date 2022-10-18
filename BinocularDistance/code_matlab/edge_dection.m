I = imread("Left11.bmp");
thresh = graythresh(I);     %自动确定二值化阈值
J = imbinarize(I,thresh);
figure
imshow(J)
I = rgb2gray(I);
Edge_canny = edge(I,'canny');
Edge_sobel = edge(I,'sobel');
figure
imshow(Edge_canny)
figure
imshow(Edge_sobel)
