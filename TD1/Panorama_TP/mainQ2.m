% Charger l'image
ima = double(imread('Amst-2.jpg'))/255;
imb = double(imread('Amst-3.jpg'))/255;

% Afficher l'image
figure(1);
imagesc(ima);

figure(2);
imagesc(imb);

% Choix des points
p1a = [397 185];
p2a = [391 144];
p3a = [406 211];
p4a = [472 138];
p5a = [494 204];
p6a = [497 128];
p7a = [475 226];

p1b = [15 184];
p2b = [9 141];
p3b = [24 211];
p4b = [92 140];
p5b = [110 202];
p6b = [116 130];
p7b = [94 226];

% Matrice PtO
PtO = [ p1a(1) p2a(1) p3a(1) p4a(1) p5a(1) p6a(1) p7a(1); ...
        p1a(2) p2a(2) p3a(2) p4a(2) p5a(2) p6a(2) p7a(2); ...
        1 1 1 1 1 1 1];
% Matrice PtD
PtD = [ p1b(1) p2b(1) p3b(1) p4b(1) p5b(1) p6b(1) p7b(1); ...
        p1b(2) p2b(2) p3b(2) p4b(2) p5b(2) p6b(2) p7b(2); ...
        1 1 1 1 1 1 1];

%homography of the images
Ha = homography2d(PtO,PtD);

Hb = homography2d(PtD,PtD);

bbox = [-710 500 -130 440];
im_warpeda = vgg_warp_H(ima, Ha, 'linear', bbox);
im_warpedb = vgg_warp_H(imb, Hb, 'linear', bbox);

im_fused = max(im_warpeda, im_warpedb);

figure(3);
imagesc(im_fused);
