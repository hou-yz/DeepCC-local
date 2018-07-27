#include "mex.h"
#include <string.h>
#include <math.h>
#include <vector>
/*
 * Usage:
 *   l = pure_potts_icm_iter_mex(W, l);
 *
 * Solve iteratively for 
 *   
 *
 * Use Gauss - Seidel iterations (i.e., use updated values in current iteration)
 *
 * W is sparse NxN real matrix 
 * l is double Nx1 vector of labels 
 *
 *
 * compile using:
 * >> mex -O -largeArrayDims pure_potts_icm_iter_mex.cpp
 */

#line   __LINE__  "pure_potts_icm_iter_mex"

#define     STR(s)      #s  
#define     ERR_CODE(a,b)   a ":" "line_" STR(b)


// INPUTS
enum {
    WIN = 0,
    LIN,
    NIN 
};

// OUTPUTS
enum {
    LOUT = 0,
    NOUT
};
template<typename T>
inline
T max(const T& a, const T& b)
{
    return (a>b)?a:b;
}

template<typename T>
inline
mwIndex AdjustCol(const typename std::vector<T>& buffer)        
{
    // what is the maximal entry?
    mwIndex mi = buffer.size()+1;
    T mx = 0;
    for (mwIndex k = 0 ; k < buffer.size() ; k++ ) {
        if ( buffer[k] > mx ) {
            mx = buffer[k];
            mi = k+1; // 1-based matlab index/label
        }
    }  
    return mi;
}

void
mexFunction(
    int nout,
    mxArray* pout[],
    int nin,
    const mxArray* pin[])
{
    if ( nin != NIN )
         mexErrMsgIdAndTxt(ERR_CODE(__FILE__, __LINE__),"must have %d inputs", NIN);
    if (nout==0)
        return;
    if (nout != NOUT )
         mexErrMsgIdAndTxt(ERR_CODE(__FILE__, __LINE__),"must have exactly %d output", NOUT);
    
    if ( mxIsComplex(pin[WIN]) || !mxIsSparse(pin[WIN]) )
        mexErrMsgIdAndTxt(ERR_CODE(__FILE__, __LINE__),"W must be real sparse matrix");
    
    if ( mxIsComplex(pin[LIN]) || mxIsSparse(pin[LIN]) || mxGetClassID(pin[LIN]) != mxDOUBLE_CLASS )
        mexErrMsgIdAndTxt(ERR_CODE(__FILE__, __LINE__),"l must be real full double vector");
    
  
    if ( mxGetN(pin[WIN]) != mxGetM(pin[WIN]) || mxGetN(pin[WIN]) != mxGetNumberOfElements(pin[LIN]) || 
            mxGetNumberOfDimensions(pin[WIN])!=2 )
        mexErrMsgIdAndTxt(ERR_CODE(__FILE__, __LINE__),"matrix - vector dimensions mismatch");

    // allocate output vector
    mwSize N = mxGetN(pin[WIN]); // n and m are equal - W is square
    
    
    pout[0] = mxCreateDoubleMatrix(N, 1, mxREAL);
    double* pl = mxGetPr(pout[0]);
    
    double* pr = mxGetPr(pin[LIN]);
    memcpy(pl, pr, N*sizeof(double));
    
    /* how many labels in initial guess? */
    mwIndex nl=0;
    for ( mwIndex ii(0); ii < N ; ii++ )        
        nl = max<mwIndex>(nl, pl[ii]);
    
    
//    mexPrintf("init has %d labels\n", nl);
    
    
    /* computation starts */
    pr = mxGetPr(pin[WIN]);
    mwIndex* pir = mxGetIr(pin[WIN]);
    mwIndex* pjc = mxGetJc(pin[WIN]);
 
    // compute each column of u' and then make a decision about it 
    // before moving on to the next columns
    for (mwSize col=0; col< N; col++)  {
        
        std::vector<double> buffer(nl,0); // start a vector of zeros the size of number of labels
        
            
        
        // perform sparse multiplication
        for (mwIndex ri = pjc[col] ; // starting row index
        ri < pjc[col+1]  ; // stopping row index
        ri++)  {
            if ( col != pir[ri] ) {
                // only off-diagonal elements are participating
//                mexPrintf("\t col=%d, row=%d, l[row]=%.0f\n", col, pir[ri], pl[ pir[ ri ] ]-1 );
                buffer[ pl[ pir[ri] ]-1 ] += pr[ri];
            }
            // pir[ri] -> current row
            // col -> current col
            // pr[ri] -> W[pir[ri], col]
            
        }
        
        
//        mexPrintf("col %d: Buffer has %d elements, buffer[0]=%.2f\n", col, buffer.size(), buffer[0]);
        
        pl[col] = AdjustCol(buffer); // make a decision for this pixel
        if ( pl[col] > nl )
            nl = pl[col];        
        
    }
}

