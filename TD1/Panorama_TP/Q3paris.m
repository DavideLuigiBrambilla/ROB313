% Charger l'image
ima = double(imread('paris_a.jpg'))/255;
imb = double(imread('paris_b.jpg'))/255;
imc = double(imread('paris_c.jpg'))/255;

% Afficher l'image
figure(1);
imagesc(ima);

figure(2);
imagesc(imb);

figure(3);
imagesc(imc);

% Transformation img 1 et 2
% points image a
p1a = [527 331];
p2a = [595 340];
p3a = [594 251];
p4a = [518 346];
p5a = [440 270];
p6a = [508 292];
p7a = [569 290];

% points image b
p1b = [249 330];
p2b = [303 336];
p3b = [306 258];
p4b = [238 343];
p5b = [166 270];
p6b = [232 294];
p7b = [284 292];

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
p1b = [351 246];
p2b = [294 262];
p3b = [553 211];
p4b = [520 347];
p5b = [249 330];
p6b = [461 218];
p7b = [421 316];

%points image c
p1c = [145 224];
p2c = [76 239];
p3c = [340 202];
p4c = [306 325];
p5c = [16 315];
p6c = [259 203];
p7c = [215 297];

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

bbox = [-700 1000 0 600];

im_warpedb = vgg_warp_H(imb, Hb, 'linear', bbox);
im_warpedc = vgg_warp_H(imc, Hc, 'linear', bbox);
im_warpeda = vgg_warp_H(ima, Ha, 'linear', bbox);


im_fused = max(im_warpedb, im_warpedc);
im_fused = max(im_fused, im_warpeda);

figure(4);
imagesc(im_fused);
