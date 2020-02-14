% Charger l'image
ima = double(imread('Amst-1.jpg'))/255;
imb = double(imread('Amst-2.jpg'))/255;
imc = double(imread('Amst-3.jpg'))/255;

% Afficher l'image
figure(1);
imagesc(ima);

figure(2);
imagesc(imb);

figure(3);
imagesc(imc);

% Transformation img 1 et 2
%points image a
p1b = [60 250];
p2b = [30 222];
p3b = [40 137];
p4b = [7 159];
p5b = [15 187];
p6b = [45 239];
p7b = [17 213];

%points image b
p1a = [496 247];
p2a = [465 220];
p3a = [473 135];
p4a = [440 157];
p5a = [448 186];
p6a = [481 238];
p7a = [451 210];

% Matrice PtO
PtOab = [p1a(1) p2a(1) p3a(1) p4a(1) p5a(1) p6a(1) p7a(1); ...
        p1a(2) p2a(2) p3a(2) p4a(2) p5a(2) p6a(2) p7a(2); ...
        1 1 1 1 1 1 1];
% Matrice PtD
PtDab = [p1b(1) p2b(1) p3b(1) p4b(1) p5b(1) p6b(1) p7b(1); ...
        p1b(2) p2b(2) p3b(2) p4b(2) p5b(2) p6b(2) p7b(2); ...
        1 1 1 1 1 1 1];
% Transformation img 2 et 3
%points image b
p1b = [397 185];
p2b = [391 144];
p3b = [406 211];
p4b = [472 138];
p5b = [494 204];
p6b = [497 128];
p7b = [475 226];

%points image c
p1c = [15 184];
p2c = [9 141];
p3c = [24 211];
p4c = [92 140];
p5c = [110 202];
p6c = [116 130];
p7c = [94 226];

% Matrice PtO
PtObc = [ p1c(1) p2c(1) p3c(1) p4c(1) p5c(1) p6c(1) p7c(1); ...
        p1c(2) p2c(2) p3c(2) p4c(2) p5c(2) p6c(2) p7c(2); ...
        1 1 1 1 1 1 1];
% Matrice PtD
PtDbc = [ p1b(1) p2b(1) p3b(1) p4b(1) p5b(1) p6b(1) p7b(1); ...
        p1b(2) p2b(2) p3b(2) p4b(2) p5b(2) p6b(2) p7b(2); ...
        1 1 1 1 1 1 1];

%homography of the images
Ha = homography2d(PtOab,PtDab);

Hb = homography2d(PtObc,PtObc);

Hc = homography2d(PtObc,PtDbc);

bbox = [-400 1000 0 400];

im_warpedb = vgg_warp_H(imb, Hb, 'linear', bbox);
im_warpedc = vgg_warp_H(imc, Hc, 'linear', bbox);
im_warpeda = vgg_warp_H(ima, Ha, 'linear', bbox);


im_fused = max(im_warpedb, im_warpedc);
im_fused = max(im_fused, im_warpeda);

figure(4);
imagesc(im_fused);
