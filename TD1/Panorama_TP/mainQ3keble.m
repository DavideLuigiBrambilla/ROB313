clear all
clc
% Charger l'image
ima = double(imread('keble_a.jpg'))/255;
imb = double(imread('keble_b.jpg'))/255;
imc = double(imread('keble_c.jpg'))/255;

% Afficher l'image
figure(1);
imagesc(ima);

figure(2);
imagesc(imb);

figure(3);
imagesc(imc);

% Transformation img 1 et 2
%points image a
p1b = [324 180];
p2b = [322 340];
p3b = [312 52];
p4b = [655 155];
p5b = [680 360];
p6b = [610 70];
p7b = [541 251];

%points image b
p1a = [31 168];
p2a = [25 340];
p3a = [20 34];
p4a = [366 161];
p5a = [384 359];
p6a = [324 79];
p7a = [254 254];

% Matrice PtO
PtOab = [p1b(1) p2b(1) p3b(1) p4b(1) p5b(1) p6b(1) p7b(1); ...
        p1b(2) p2b(2) p3b(2) p4b(2) p5b(2) p6b(2) p7b(2); ...
        1 1 1 1 1 1 1];
% Matrice PtD
PtDab = [p1a(1) p2a(1) p3a(1) p4a(1) p5a(1) p6a(1) p7a(1); ...
        p1a(2) p2a(2) p3a(2) p4a(2) p5a(2) p6a(2) p7a(2); ...
        1 1 1 1 1 1 1];
% Transformation img 2 et 3
%points image b
p1b = [74 362];
p2b = [346 174];
p3b = [347 338];
p4b = [211 126];
p5b = [306 339];
p6b = [228 27];
p7b = [57 103];

%points image c
p1c = [384 360];
p2c = [653 165];
p3c = [657 334];
p4c = [511 126];
p5c = [613 336];
p6c = [523 21];
p7c = [364 116];

% Matrice PtO
PtObc = [ p1b(1) p2b(1) p3b(1) p4b(1) p5b(1) p6b(1) p7b(1); ...
        p1b(2) p2b(2) p3b(2) p4b(2) p5b(2) p6b(2) p7b(2); ...
        1 1 1 1 1 1 1];
    
% Matrice PtD
PtDbc = [ p1c(1) p2c(1) p3c(1) p4c(1) p5c(1) p6c(1) p7c(1); ...
        p1c(2) p2c(2) p3c(2) p4c(2) p5c(2) p6c(2) p7c(2); ...
        1 1 1 1 1 1 1];


%homography of the images
Ha = homography2d(PtOab,PtDab);

Hb = homography2d(PtObc,PtObc);

Hc = homography2d(PtObc,PtDbc);

bbox = [-380 1000 -80 630];

im_warpedb = vgg_warp_H(imb, Hb, 'linear', bbox);
im_warpeda = vgg_warp_H(ima, Ha, 'linear', bbox);
im_warpedc = vgg_warp_H(imc, Hc, 'linear', bbox);


im_fused = max(im_warpeda, im_warpedb);
im_fused = max(im_fused, im_warpedc);

figure(4);
imagesc(im_fused);
