// Imagine++ project
// Project:  Fundamental
// Author:   Pascal Monasse
// Date:     2013/10/08

#include "./Imagine/Features.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace Imagine;
using namespace std;

static const float BETA = 0.01f; // Probability of failure

struct Match {
    float x1, y1, x2, y2;
};

// Display SIFT points and fill vector of point correspondences
void algoSIFT(Image<Color,2> I1, Image<Color,2> I2,
              std::vector<Match>& matches) {
    // Find interest points
    SIFTDetector D;
    D.setFirstOctave(-1);
    Array<SIFTDetector::Feature> feats1 = D.run(I1);
    drawFeatures(feats1, Coords<2>(0,0));
    cout << "Im1: " << feats1.size() << flush;
    Array<SIFTDetector::Feature> feats2 = D.run(I2);
    drawFeatures(feats2, Coords<2>(I1.width(),0));
    cout << " Im2: " << feats2.size() << flush;

    const double MAX_DISTANCE = 100.0*100.0;
    for(size_t i=0; i < feats1.size(); i++) {
        SIFTDetector::Feature f1=feats1[i];
        for(size_t j=0; j < feats2.size(); j++) {
            double d = squaredDist(f1.desc, feats2[j].desc);
            if(d < MAX_DISTANCE) {
                Match m;
                m.x1 = f1.pos.x();
                m.y1 = f1.pos.y();
                m.x2 = feats2[j].pos.x();
                m.y2 = feats2[j].pos.y();
                matches.push_back(m);
            }
        }
    }
}

FMatrix<float,3,3> algo8Points(const vector<Match>& matches,
                               const vector<int>& inliers) {
    const float NORM=1.0E-3f; // Normalization
    Matrix<float> A(max(9,(int)inliers.size()), 9);
    for(int i=0; i<(int)inliers.size(); i++) {
        float x1 = NORM*matches[inliers[i]].x1;
        float y1 = NORM*matches[inliers[i]].y1;
        float x2 = NORM*matches[inliers[i]].x2;
        float y2 = NORM*matches[inliers[i]].y2;
        A(i,0) = x1*x2;
        A(i,1) = x1*y2;
        A(i,2) = x1;
        A(i,3) = y1*x2;
        A(i,4) = y1*y2;
        A(i,5) = y1;
        A(i,6) = x2;
        A(i,7) = y2;
        A(i,8) = 1;
    }
    // Add equation 0 X = 0 to get at least 9 equations
    if(inliers.size()<9)
        for(int i=0; i<9; i++)
            A(8,i) = 0;
    // SVD
    Matrix<float> U, Vt; Vector<float> S;
    svd(A, U, S, Vt);
    // Copy last column of V = last row of Vt
    FMatrix<float,3,3> F;
    F(0,0) = Vt(Vt.nrow()-1, 0);
    F(0,1) = Vt(Vt.nrow()-1, 1);
    F(0,2) = Vt(Vt.nrow()-1, 2);
    F(1,0) = Vt(Vt.nrow()-1, 3);
    F(1,1) = Vt(Vt.nrow()-1, 4);
    F(1,2) = Vt(Vt.nrow()-1, 5);
    F(2,0) = Vt(Vt.nrow()-1, 6);
    F(2,1) = Vt(Vt.nrow()-1, 7);
    F(2,2) = Vt(Vt.nrow()-1, 8);
    // Denormalization
    FMatrix<float,3,3> N;
    N.fill(0);
    N(0,0)=N(1,1)=NORM;
    N(2,2)=1;
    return transpose(N)*F*N;
}

// RANSAC algorithm to compute F from point matches (8-point algorithm).
// Parameter matches is filtered to keep only inliers as output.
FMatrix<float,3,3> computeF(vector<Match>& matches) {
    const float distMax = 1.5f; // Pixel error for inlier/outlier discrimination
    int Niter=100000; // Adjusted dynamically
    FMatrix<float,3,3> bestF;
    vector<int> bestInliers;
    // --------------- TODO ------------
    // DO NOT FORGET NORMALIZATION OF POINTS
    for(int i=0; i<Niter; i++) {
        vector<int> inliers;
        for(int j=0; j<8; j++)
            inliers.push_back(rand()%matches.size());
        FMatrix<float,3,3> F = algo8Points(matches, inliers);
        // Find inliers
        inliers.clear();
        for(int i=0; i<(int)matches.size(); i++) {
            FVector<float,3> x (matches[i].x1, matches[i].y1, 1);
            FVector<float,3> xp(matches[i].x2, matches[i].y2, 1);
            xp = F*xp;
            float d2 = (x*xp)*(x*xp)/(xp[0]*xp[0]+xp[1]*xp[1]);
            if(d2 < distMax*distMax)
                inliers.push_back(i);
        }
        if(inliers.size() > bestInliers.size()) {
            bestInliers = inliers;
            float denom = log(1-pow(inliers.size()/(float)matches.size(), 8));
            Niter = (int)ceil(log(BETA)/denom);
        }
    }
    cout << "inliers: " << bestInliers.size() << ", Niter: " << Niter << endl;
    bestF = algo8Points(matches, bestInliers);
    
    // Updating matches with inliers only
    vector<Match> all=matches;
    matches.clear();
    for(size_t i=0; i<bestInliers.size(); i++)
        matches.push_back(all[bestInliers[i]]);
    return bestF;
}

// Expects clicks in one image and show corresponding line in other image.
// Stop at right-click.
void displayEpipolar(Image<Color> I1, Image<Color> I2,
                     const FMatrix<float,3,3>& F) {
    while(true) {
        int x,y;
        if(getMouse(x,y) == 3)
            break;
        // --------------- TODO ------------
        bool right = (x>=I1.width());
        drawCircle(x,y,4, RED);
        FVector<float,3> l((float)x,(float)y,1);
        if(right) {
            l[0] -= I1.width();
            l = F*l;
            float y1 = -l[2]/l[1]; // left intersection
            float y2 = -(l[0]*I1.width()+l[2])/l[1]; // right intersection
            drawLine(0,(int)y1,I1.width(),(int)y2, GREEN);
        } else {            
            l = transpose(F)*l;
            float y1 = -l[2]/l[1]; // left intersection
            float y2 = -(l[0]*I2.width()+l[2])/l[1]; // right intersection
            drawLine(I1.width(),(int)y1,I1.width()+I2.width(),(int)y2, GREEN);
        } 
    }
}

int main(int argc, char* argv[])
{
    srand((unsigned int)time(0));

    const char* s1 = argc>1? argv[1]: srcPath("im1.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("im2.jpg");

    // Load and display images
    Image<Color,2> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load images" << endl;
        return 1;
    }
    int w = I1.width();
    openWindow(2*w, I1.height());
    display(I1,0,0);
    display(I2,w,0);

    vector<Match> matches;
    algoSIFT(I1, I2, matches);
    cout << " matches: " << matches.size() << endl;
    click();
    
    FMatrix<float,3,3> F = computeF(matches);
    cout << "F="<< endl << F;

    // Redisplay with matches
    display(I1,0,0);
    display(I2,w,0);
    for(size_t i=0; i<matches.size(); i++) {
        Color c(rand()%256,rand()%256,rand()%256);
        fillCircle(matches[i].x1+0, matches[i].y1, 2, c);
        fillCircle(matches[i].x2+w, matches[i].y2, 2, c);        
    }
    click();

    // Redisplay without SIFT points
    display(I1,0,0);
    display(I2,w,0);
    displayEpipolar(I1, I2, F);

    endGraphics();
    return 0;
}
