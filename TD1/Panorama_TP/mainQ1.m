clear all
clc
% Charger l'image
imrgb = double(imread('Pompei.jpg'))/255;

% Afficher l'image
figure(1);
imagesc(imrgb);

% Choix des points - the corners of the image
p1 = [142 45];
p2 = [356 45];
p3 = [111 255];
p4 = [371 261];

% Matrice PtO
PtO = [ p1(1)   p2(1)   p3(1)   p4(1); ...
        p1(2)   p2(2)   p3(2)   p4(2); ...
        1       1       1       1]
% Matrice PtD
PtD = [ p1(1)   p2(1)   p1(1)   p2(1); ...  % Choix des points pour transformer
        p1(2)   p2(2)   p3(2)   p4(2); ...  % l'image du mosaic dans une image
        1       1       1       1]          % carré

H = homography2d(PtO,PtD);

% Afficher l'image
figure(2);
imagesc(vgg_warp_H(imrgb,H));