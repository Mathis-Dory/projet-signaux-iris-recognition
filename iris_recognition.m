close all;
clear;
first_img = imread("database/001/01.bmp");
second_img = imread("database/003/01.bmp");

subplot(3,2,1);
imshow(first_img);
title('Image de référence');

subplot(3,2,2);
imshow(second_img);
title('Image de comparaison');


% Image RGB à Gris
first_img_gray = im2gray(first_img);
second_img_gray = im2gray(second_img);

%Filtre de Canny
subplot(3,2,3);
first_canny = edge(first_img_gray, 'canny',[0.1 0.20] , 10);
imshow(first_canny);

subplot(3,2,4);
second_canny = edge(second_img_gray, 'canny', [0.1 0.20], 10);
imshow(second_canny);

%Transformée de Hough

radii = 40:1:160;
h = circle_hough(first_canny, radii, 'same');
peaks = circle_houghpeaks(h, radii, 'nhoodxy', 15, 'nhoodr', 21, 'npeaks', 2);
subplot(3,2,5);
imshow(first_img);
hold on;
for peak = peaks
    [x, y] = circlepoints(peak(3));
    plot(x+peak(1), y+peak(2), 'g-');
end
hold off

radii = 40:1:160;
h = circle_hough(second_canny, radii, 'same');
peaks = circle_houghpeaks(h, radii, 'nhoodxy', 15, 'nhoodr', 21, 'npeaks', 2);
subplot(3,2,6);
imshow(second_img);
hold on;
for peak = peaks
    [x, y] = circlepoints(peak(3));
    plot(x+peak(1), y+peak(2), 'g-');
end
hold off
