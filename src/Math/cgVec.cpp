#include "cgVec.hpp"
#include <assert.h>
#include <algorithm>

#ifdef _WIN64
    #define USE_XMM_SIMD
#endif

#ifdef USE_XMM_SIMD //use XMM simd intrinsics for speeding up matrix mult, otherwise just use the non SIMD version
    #include <xmmintrin.h>
#endif

cgVec2::cgVec2(float x,float y){
    this->x = x;
    this->y = y;
}
cgVec2::cgVec2(const cgVec2& vector){
    x = vector.x;
    y = vector.y;
}
cgVec2 cgVec2::operator+(const cgVec2& vector){
    return cgVec2(this->x+vector.x, this->y+vector.y);
}
cgVec2 cgVec2::operator-(const cgVec2& vector){
    return cgVec2(this->x-vector.x, this->y-vector.y);
}
cgVec2 cgVec2::operator*(const float& scalar){
    return cgVec2(this->x*scalar,this->y*scalar);
}
float cgVec2::innerProd(const cgVec2& vector){
    return this->x*vector.x + this->y*vector.y;
}
cgVec2 cgVec2::normalized(){
    float norm = sqrt(x*x+y*y);
    return cgVec2(x/norm,y/norm);
}
cgVec2 cgVec2::rotatedBy(float radians){
    cgMat2 R = cgMat2(cos(radians), -1*sin(radians), sin(radians), cos(radians));
    return R*(*this);
}
void cgVec2::rotateBy(float radians){
    cgMat2 R = cgMat2(cos(radians), -1*sin(radians), sin(radians), cos(radians));
    cgVec2 temp = R*(*this);
    this->x = temp.x;
    this->y = temp.y;
}

float dot(cgVec2 v1,cgVec2 v2){
    return v1.x*v2.x + v1.y*v2.y;
}

cgMat2::cgMat2(float r1c1, float r1c2,
               float r2c1, float r2c2){
    this->data[0] = r1c1;
    this->data[1] = r1c2;
    this->data[2] = r2c1;
    this->data[3] = r2c2;
}
float cgMat2::at(int row, int col){
    return data[row*2 + col];
}
cgVec2 cgMat2::operator*(const cgVec2& vector){
    return cgVec2(data[0]*vector.x+data[1]*vector.y, data[2]*vector.x+data[3]*vector.y);
}

cg2DFloatArray::cg2DFloatArray(int numRows, int numCols){
    data = new float*[numRows];
    for (int i = 0; i < numRows; i++){
        data[i] = new float[numCols];
    }
    this->numCols = numCols;
    this->numRows = numRows;
}
cg2DFloatArray::~cg2DFloatArray(){
    for(int i = 0; i < numRows;i++){
        delete[] data[i];
    }
    delete [] data;
}
float cg2DFloatArray::at(int row, int col){
    return data[row][col];
}
void cg2DFloatArray::set(int row, int col, float to){
    this->data[row][col] = to;
}
float cg2DFloatArray::max(){
    float max = data[0][0];
    for(int i = 0; i < numRows; i++){
        for(int j = 0; j < numCols; j++){
            if (data[i][j] > max){
                max = data[i][j];
            }
        }
    }
    return max;
}
float cg2DFloatArray::min(){
    float min = data[0][0];
    for(int i = 0; i < numRows; i++){
        for(int j = 0; j < numCols; j++){
            if (data[i][j] < min){
                min = data[i][j];
            }
        }
    }
    return min;
}
cgVec3 cgVec3::crossProd(const cgVec3& vector){
    cgVec3 output = cgVec3(y*vector.z - z*vector.y,z*vector.x - x*vector.z,x*vector.y-y*vector.x);
    return output;
}
void cgVec3::normalize(){
    float r = norm();
    x = x/r;
    y = y/r;
    z = z/r;
}
float cgVec3::norm(){
    return sqrt(x*x+y*y+z*z);
}

void cgVec3::getOrthogonalVectors(cgVec3& target1, cgVec3& target2){
    //find smallest dot product between standard basis vectors
    cgVec3 basis[3] = {cgVec3(1,0,0),cgVec3(0,1,0),cgVec3(0,0,1)};
    float minDot = std::numeric_limits<float>::infinity();
    int minBasis = 0;

    for(int i = 0; i < 3; i++){
        float d = this->dot(basis[i]);
        if(d < minDot){
            minDot=d;
            minBasis = i;
        }
    }

    cgVec3 normal1 = cross(*this,basis[minBasis]);
    cgVec3 normal2 = cross(*this, normal1);
    target1 = normal1;
    target2 = normal2;
}

cgVec3 cgVec3::getOrthogonalVector(){
    cgVec3 basis[3] = {cgVec3(1,0,0),cgVec3(0,1,0),cgVec3(0,0,1)};
    float minDot = std::numeric_limits<float>::infinity();
    int minBasis = 0;

    for(int i = 0; i < 3; i++){
        float d = this->dot(basis[i]);
        if(d < minDot){
            minDot=d;
            minBasis = i;
        }
    }

    cgVec3 normal1 = cross(*this,basis[minBasis]);
    return normal1;
}

cgVec3 cgVec3::matMult4(const cgMat4& mat){
    cgVec3 output = cgVec3();
    float sum = 0;
    for (int i = 0; i<3; i++){
        sum = 0;
        for (int j = 0; j < 3; j++){
            sum+=at(j)*mat.at(i,j);
        }
        sum+= mat.at(i, 3);
        output.set(i, sum);
    }
    return output;
}

cgVec3 cgVec3::ExtractPosition(const cgMat4& matrix) {
    return cgVec3(matrix.at(0, 3), matrix.at(1, 3), matrix.at(2, 3));
}

cgVec3 cgVec3::ExtractDirection(const cgMat4& matrix) {
    cgVec3 direction(matrix.at(0, 2), matrix.at(1, 2), matrix.at(2, 2));
    direction.normalize();
    return direction;
}




cgMat4::cgMat4(float xx,float yx,float zx,float wx,
               float xy,float yy,float zy,float wy,
               float xz,float yz,float zz,float wz,
               float xw,float yw,float zw, float ww){
    data[0] = xx; //00
    data[1] = xy; //10
    data[2] = xz; //20
    data[3] = xw; //30
    
    data[4] = yx; 
    data[5] = yy; //11
    data[6] = yz;
    data[7] = yw;
    
    data[8] =  zx;
    data[9] =  zy;
    data[10] = zz;
    data[11] = zw;

    data[12] = wx;
    data[13] = wy;
    data[14] = wz;
    data[15] = ww;
}
void cgMat4::removeTranslation(){
    //data[3]=0; data[7]=0; data[11]=0;
    //data[12]=0; data[13]=0; //data[14]=0; //data[15]=0;
//    (1, 0, 0, Tx,
//     0, 1, 0, Ty,
//     0, 0, 1, Tz,
//     0, 0, 0, 1);
}
cgMat4 cgMat4::operator*(const cgMat4& mat) const{
    cgMat4 output = cgMat4();
    float sum;

    #ifdef USE_XMM_SIMD
    __m128 row1 = _mm_load_ps(&this->data[0]);
    __m128 row2 = _mm_load_ps(&this->data[4]);
    __m128 row3 = _mm_load_ps(&this->data[8]);
    __m128 row4 = _mm_load_ps(&this->data[12]);
    for(int i=0; i<4; i++) {
        __m128 brod1 = _mm_set1_ps(mat.data[4*i + 0]);
        __m128 brod2 = _mm_set1_ps(mat.data[4*i + 1]);
        __m128 brod3 = _mm_set1_ps(mat.data[4*i + 2]);
        __m128 brod4 = _mm_set1_ps(mat.data[4*i + 3]);
        __m128 row = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(brod1, row1),
                        _mm_mul_ps(brod2, row2)),
                    _mm_add_ps(
                        _mm_mul_ps(brod3, row3),
                        _mm_mul_ps(brod4, row4)));
            _mm_store_ps(&output.data[4*i], row);
    }
    return output;
    #else
    for(int i = 0; i<4;i++){
        for(int j = 0;j<4;j++){
            sum = 0;
            for(int k = 0; k < 4;k++){
                sum = sum + at(i, k)*mat.at(k,j);
            }
            output.set(i, j, sum);
        }
    }
    return output;
    #endif
}

float cgMat4::determinant(){

    return this->at(0,0)*this->at(1,1)*this->at(2,2)*this->at(3,3) - this->at(0,0)*this->at(1,1)*this->at(2,3)*this->at(3,2) + this->at(0,0)*this->at(1,2)*this->at(2,3)*this->at(3,1) - this->at(0,0)*this->at(1,2)*this->at(2,1)*this->at(3,3)
            + this->at(0,0)*this->at(1,3)*this->at(2,1)*this->at(3,2) - this->at(0,0)*this->at(1,3)*this->at(2,2)*this->at(3,1) - this->at(0,1)*this->at(1,2)*this->at(2,3)*this->at(3,0) + this->at(0,1)*this->at(1,2)*this->at(2,0)*this->at(3,3)
            - this->at(0,1)*this->at(1,3)*this->at(2,0)*this->at(3,2) + this->at(0,1)*this->at(1,3)*this->at(2,2)*this->at(3,0) - this->at(0,1)*this->at(1,0)*this->at(2,2)*this->at(3,3) + this->at(0,1)*this->at(1,0)*this->at(2,3)*this->at(3,2)
            + this->at(0,2)*this->at(1,3)*this->at(2,0)*this->at(3,1) - this->at(0,2)*this->at(1,3)*this->at(2,1)*this->at(3,0) + this->at(0,2)*this->at(1,0)*this->at(2,1)*this->at(3,3) - this->at(0,2)*this->at(1,0)*this->at(2,3)*this->at(3,1)
            + this->at(0,2)*this->at(1,1)*this->at(2,3)*this->at(3,0) - this->at(0,2)*this->at(1,1)*this->at(2,0)*this->at(3,3) - this->at(0,3)*this->at(1,0)*this->at(2,1)*this->at(3,2) + this->at(0,3)*this->at(1,0)*this->at(2,2)*this->at(3,1)
            - this->at(0,3)*this->at(1,1)*this->at(2,2)*this->at(3,0) + this->at(0,3)*this->at(1,1)*this->at(2,0)*this->at(3,2) - this->at(0,3)*this->at(1,2)*this->at(2,0)*this->at(3,1) + this->at(0,3)*this->at(1,2)*this->at(2,1)*this->at(3,0);
}

cgMat4 cgMat4::inverse(){
    // Compute the reciprocal determinant
    float det = determinant();

    if(det == 0.0f)
    {
        assert(0);
        return *this;
    }

    float invdet = 1.0f / det;

    cgMat4 res;
    res.set(0,0, invdet  * (at(1,1) * (at(2,2) * at(3,3) - at(2,3) * at(3,2)) + at(1,2) *
                  (at(2,3) * at(3,1) - at(2,1) * at(3,3)) + at(1,3) * (at(2,1) * at(3,2) - at(2,2) * at(3,1))));
    res.set(0,1, -invdet * (at(0,1) * (at(2,2) * at(3,3) - at(2,3) * at(3,2)) + at(0,2) *
                  (at(2,3) * at(3,1) - at(2,1) * at(3,3)) + at(0,3) * (at(2,1) * at(3,2) - at(2,2) * at(3,1))));
    res.set(0,2, invdet  * (at(0,1) * (at(1,2) * at(3,3) - at(1,3) * at(3,2)) + at(0,2) *
                  (at(1,3) * at(3,1) - at(1,1) * at(3,3)) + at(0,3) * (at(1,1) * at(3,2) - at(1,2) * at(3,1))));
    res.set(0,3, -invdet * (at(0,1) * (at(1,2) * at(2,3) - at(1,3) * at(2,2)) + at(0,2) *
                  (at(1,3) * at(2,1) - at(1,1) * at(2,3)) + at(0,3) * (at(1,1) * at(2,2) - at(1,2) * at(2,1))));
    res.set(1,0, -invdet * (at(1,0) * (at(2,2) * at(3,3) - at(2,3) * at(3,2)) + at(1,2) *
                  (at(2,3) * at(3,0) - at(2,0) * at(3,3)) + at(1,3) * (at(2,0) * at(3,2) - at(2,2) * at(3,0))));
    res.set(1,1, invdet  * (at(0,0) * (at(2,2) * at(3,3) - at(2,3) * at(3,2)) + at(0,2) *
                  (at(2,3) * at(3,0) - at(2,0) * at(3,3)) + at(0,3) * (at(2,0) * at(3,2) - at(2,2) * at(3,0))));
    res.set(1,2, -invdet * (at(0,0) * (at(1,2) * at(3,3) - at(1,3) * at(3,2)) + at(0,2) *
                  (at(1,3) * at(3,0) - at(1,0) * at(3,3)) + at(0,3) * (at(1,0) * at(3,2) - at(1,2) * at(3,0))));
    res.set(1,3, invdet  * (at(0,0) * (at(1,2) * at(2,3) - at(1,3) * at(2,2)) + at(0,2) *
                  (at(1,3) * at(2,0) - at(1,0) * at(2,3)) + at(0,3) * (at(1,0) * at(2,2) - at(1,2) * at(2,0))));
    res.set(2,0, invdet  * (at(1,0) * (at(2,1) * at(3,3) - at(2,3) * at(3,1)) + at(1,1) *
                  (at(2,3) * at(3,0) - at(2,0) * at(3,3)) + at(1,3) * (at(2,0) * at(3,1) - at(2,1) * at(3,0))));
    res.set(2,1, -invdet * (at(0,0) * (at(2,1) * at(3,3) - at(2,3) * at(3,1)) + at(0,1) *
                  (at(2,3) * at(3,0) - at(2,0) * at(3,3)) + at(0,3) * (at(2,0) * at(3,1) - at(2,1) * at(3,0))));
    res.set(2,2, invdet  * (at(0,0) * (at(1,1) * at(3,3) - at(1,3) * at(3,1)) + at(0,1) *
                  (at(1,3) * at(3,0) - at(1,0) * at(3,3)) + at(0,3) * (at(1,0) * at(3,1) - at(1,1) * at(3,0))));
    res.set(2,3, -invdet * (at(0,0) * (at(1,1) * at(2,3) - at(1,3) * at(2,1)) + at(0,1) *
                  (at(1,3) * at(2,0) - at(1,0) * at(2,3)) + at(0,3) * (at(1,0) * at(2,1) - at(1,1) * at(2,0))));
    res.set(3,0, -invdet * (at(1,0) * (at(2,1) * at(3,2) - at(2,2) * at(3,1)) + at(1,1) *
                  (at(2,2) * at(3,0) - at(2,0) * at(3,2)) + at(1,2) * (at(2,0) * at(3,1) - at(2,1) * at(3,0))));
    res.set(3,1, invdet  * (at(0,0) * (at(2,1) * at(3,2) - at(2,2) * at(3,1)) + at(0,1) *
                  (at(2,2) * at(3,0) - at(2,0) * at(3,2)) + at(0,2) * (at(2,0) * at(3,1) - at(2,1) * at(3,0))));
    res.set(3,2, -invdet * (at(0,0) * (at(1,1) * at(3,2) - at(1,2) * at(3,1)) + at(0,1) *
                  (at(1,2) * at(3,0) - at(1,0) * at(3,2)) + at(0,2) * (at(1,0) * at(3,1) - at(1,1) * at(3,0))));
    res.set(3,3, invdet  * (at(0,0) * (at(1,1) * at(2,2) - at(1,2) * at(2,1)) + at(0,1) *
                            (at(1,2) * at(2,0) - at(1,0) * at(2,2)) + at(0,2) * (at(1,0) * at(2,1) - at(1,1) * at(2,0))));
    return res;
}

cgVec4 cgMat4::operator*(const cgVec4& vec) const{
    return cgVec4(  data[0]*vec.x+data[4]*vec.y+data[8]*vec.z+data[12]*vec.w,
                    data[1]*vec.x+data[5]*vec.y+data[9]*vec.z+data[13]*vec.w,
                    data[2]*vec.x+data[6]*vec.y+data[10]*vec.z+data[14]*vec.w,
                    data[3]*vec.x+data[7]*vec.y+data[11]*vec.z+data[15]*vec.w
                    );
}

cgVec3 cgMat4::operator*(const cgVec3& vec) const{
    cgVec4 temp = cgVec4(vec.x,vec.y,vec.z,1.0);
    cgVec4 m = this->operator*(temp);
    return cgVec3(m.x,m.y,m.z);
}

void cgMat4::print(){
    for(int i = 0; i<4;i++){
        std::cout << "|";
        for(int j = 0;j<4;j++){
            std::cout << at(i, j) << " ";
        }
        std::cout << "|\n";
    }
}

quat::quat(cgMat4 rm){
    // Extract quaternion from rotation matrix using the most numerically stable method
    // Based on the algorithm from "Quaternion Calculus and Fast Animation" by Ken Shoemake

    float m00 = rm.at(0,0), m01 = rm.at(0,1), m02 = rm.at(0,2);
    float m10 = rm.at(1,0), m11 = rm.at(1,1), m12 = rm.at(1,2);
    float m20 = rm.at(2,0), m21 = rm.at(2,1), m22 = rm.at(2,2);

    float tr = m00 + m11 + m22; // trace

    if (tr > 0) {
        // Use w component - most stable when trace is positive
        float S = sqrt(tr + 1.0f) * 2; // S = 4w
        this->w = S * 0.25f;
        this->v.x = (m21 - m12) / S;
        this->v.y = (m02 - m20) / S;
        this->v.z = (m10 - m01) / S;
    } else if ((m00 > m11) && (m00 > m22)) {
        // Use x component
        float S = sqrt(1.0f + m00 - m11 - m22) * 2; // S = 4x
        this->w = (m21 - m12) / S;
        this->v.x = S * 0.25f;
        this->v.y = (m01 + m10) / S;
        this->v.z = (m02 + m20) / S;
    } else if (m11 > m22) {
        // Use y component
        float S = sqrt(1.0f + m11 - m00 - m22) * 2; // S = 4y
        this->w = (m02 - m20) / S;
        this->v.x = (m01 + m10) / S;
        this->v.y = S * 0.25f;
        this->v.z = (m12 + m21) / S;
    } else {
        // Use z component
        float S = sqrt(1.0f + m22 - m00 - m11) * 2; // S = 4z
        this->w = (m10 - m01) / S;
        this->v.x = (m02 + m20) / S;
        this->v.y = (m12 + m21) / S;
        this->v.z = S * 0.25f;
    }
}

cgMat4 quat::getRotMatrix(){
    return cgMat4(1-2*(v.y*v.y+v.z*v.z), 2*(v.x*v.y-v.z*w), 2*(v.x*v.z+v.y*w), 0,
                  2*(v.x*v.y+v.z*w), 1-2*(v.x*v.x+v.z*v.z), 2*(v.y*v.z-v.x*w), 0,
                  2*(v.x*v.z-v.y*w), 2*(v.y*v.z+v.x*w), 1-2*(v.x*v.x+v.y*v.y), 0,
                  0, 0, 0, 1);
}

void quat::getEulerAngles(float& yaw, float& pitch, float& roll){
    yaw = atan2(2.0*(v.y*v.z + w*v.x), w*w - v.x*v.x - v.y*v.y + v.z*v.z);
    // Clamp to [-1, 1] to handle floating point errors
    float pitchValue = -2.0*(v.x*v.z - w*v.y);
    pitchValue = std::max(-1.0f, std::min(1.0f, pitchValue));
    pitch = asin(pitchValue);
    roll = atan2(2.0*(v.x*v.y + w*v.z), w*w + v.x*v.x - v.y*v.y - v.z*v.z);
}

cgMat4* translation(float Tx, float Ty, float Tz){
    return new cgMat4(1, 0, 0, Tx,
                      0, 1, 0, Ty,
                      0, 0, 1, Tz,
                      0, 0, 0, 1);
}

cgMat4 trans(float Tx, float Ty, float Tz){
        return cgMat4(1, 0, 0, Tx,
                      0, 1, 0, Ty,
                      0, 0, 1, Tz,
                      0, 0, 0, 1);
}

void translation(float Tx, float Ty, float Tz, cgMat4& mat){
    mat.set(0, 3, Tx);
    mat.set(1, 3, Ty);
    mat.set(2, 3, Tz);
}
cgMat4* scaling(float Sx, float Sy, float Sz){
    return new cgMat4(Sx, 0, 0, 0,
                      0, Sy, 0, 0,
                      0, 0, Sz, 0,
                      0, 0, 0, 1);
}

cgMat4 scale(float Sx, float Sy, float Sz){
        return cgMat4(Sx, 0, 0, 0,
                      0, Sy, 0, 0,
                      0, 0, Sz, 0,
                      0, 0, 0, 1);
}

cgMat4 ortho(const float left, 
             const float right, 
             const float bottom, 
             const float top, 
             const float zNear, 
             const float zFar){

    cgMat4 result;
    result.set(0,0, 2/(right-left));
    result.set(1,1, 2 / (top - bottom));
    result.set(2,2, -2 / (zFar - zNear));
    result.set(3,0, -(right + left) / (right - left));
    result.set(3,1, -(top + bottom) / (top - bottom));
    result.set(3,2, -(zFar + zNear) / (zFar - zNear));
    result.set(3,3, 1);
    return result;
}
    
cgMat4 lookAt_glm(cgVec3 eye, cgVec3 center, cgVec3 up)
{
    
    cgVec3 zaxis = (center - eye).normalized();    
    cgVec3 xaxis = (cross(zaxis, up)).normalized();
    cgVec3 yaxis = cross(xaxis, zaxis);
    zaxis = zaxis*-1;
    cgMat4 result = {
        xaxis.x, xaxis.y, xaxis.z, -dot(xaxis, eye),
        yaxis.x, yaxis.y, yaxis.z, -dot(yaxis, eye),
        zaxis.x, zaxis.y, zaxis.z, -dot(zaxis, eye),
        0, 0, 0, 1
    };

    return result;
}

cgMat4* rotation(cgVec3 axis, float theta){
    return new cgMat4(cos(theta)+axis.x*axis.x*(1-cos(theta)), axis.x*axis.y*(1-cos(theta))-axis.z*sin(theta), axis.x*axis.z*(1-cos(theta))+axis.y*sin(theta), 0,
                      
                     axis.y*axis.x*(1-cos(theta))+axis.z*sin(theta), cos(theta)+axis.y*axis.y*(1-cos(theta)), axis.y*axis.z*(1-cos(theta))-axis.x*sin(theta), 0,
                      
                     axis.z*axis.x*(1-cos(theta))-axis.y*sin(theta), axis.z*axis.y*(1-cos(theta))+axis.x*sin(theta), cos(theta)+axis.z*axis.z*(1-cos(theta)), 0,
                      
                     0, 0, 0, 1);
}

cgMat4 rot(cgVec3 axis, float theta){
    return cgMat4(cos(theta)+axis.x*axis.x*(1-cos(theta)), axis.x*axis.y*(1-cos(theta))-axis.z*sin(theta), axis.x*axis.z*(1-cos(theta))+axis.y*sin(theta), 0,
                      
                     axis.y*axis.x*(1-cos(theta))+axis.z*sin(theta), cos(theta)+axis.y*axis.y*(1-cos(theta)), axis.y*axis.z*(1-cos(theta))-axis.x*sin(theta), 0,
                      
                     axis.z*axis.x*(1-cos(theta))-axis.y*sin(theta), axis.z*axis.y*(1-cos(theta))+axis.x*sin(theta), cos(theta)+axis.z*axis.z*(1-cos(theta)), 0,
                      
                     0, 0, 0, 1);
}

cgMat4 rot(float a, float b, float g){
    return cgMat4  (cos(a)*cos(b), cos(a)*sin(b)*sin(g)-sin(a)*cos(g), cos(a)*sin(b)*cos(g)+sin(a)*sin(g), 0,
                    sin(a)*cos(b), sin(a)*sin(b)*sin(g)+cos(a)*cos(g),sin(a)*sin(b)*cos(g)-cos(a)*sin(g), 0,
                    -sin(b), cos(b)*sin(g), cos(b)*cos(g), 0,
                    0, 0, 0, 1);
}

void rotation(cgVec3 axis, float theta, cgMat4& mat){
    mat.set(0, 0, cos(theta)+axis.x*axis.x*(1-cos(theta)));
    mat.set(0, 1, axis.x*axis.y*(1-cos(theta))-axis.z*sin(theta));
    mat.set(0, 2, axis.x*axis.z*(1-cos(theta))+axis.y*sin(theta));
    mat.set(0, 3, 0);
    
    mat.set(1, 0, axis.y*axis.x*(1-cos(theta))+axis.z*sin(theta));
    mat.set(1, 1, cos(theta)+axis.y*axis.y*(1-cos(theta)));
    mat.set(1, 2, axis.y*axis.z*(1-cos(theta))-axis.x*sin(theta));
    mat.set(1, 3, 0);
    
    mat.set(2, 0, axis.z*axis.x*(1-cos(theta))-axis.y*sin(theta));
    mat.set(2, 1, axis.z*axis.y*(1-cos(theta))+axis.x*sin(theta));
    mat.set(2, 2, cos(theta)+axis.z*axis.z*(1-cos(theta)));
    mat.set(2, 3, 0);
    
    mat.set(3, 0, 0);mat.set(3, 1, 0);mat.set(3, 2, 0);mat.set(3, 3, 1);
}

cgMat4* projectionMatrix(float n, float r, float t, float f){
    return new cgMat4(n/r, 0, 0, 0,
                      0, n/t, 0, 0,
                      0, 0, -(f+n)/(f-n), -2*f*n/(f-n),
                      0, 0, -1, 0);
}

cgMat4* projectionMatrixSimple(float aspect, float fov, float n, float f){
    float f_p = 1.0/(tan(fov/2));
    return new cgMat4(f_p/aspect, 0, 0, 0,
                      0, f_p, 0, 0,
                      0, 0, (f+n)/(n-f), (2*f*n)/(n-f),
                      0, 0, -1, 0);
}

cgMat4* orthoProjection(float n, float r, float t, float f){
    return new cgMat4(1/r, 0, 0, 0,
                      0, 1/t, 0, 0,
                      0, 0, -2/(f-n), -(f+n)/(f-n),
                      0, 0, 0, 1);
}

cgMat4* orthoProjectionWithBounds(float l, float r, float b, float t, float n, float f) {
    return new cgMat4(2.0f / (r - l), 0.0f, 0.0f, -(r + l) / (r - l),
        0.0f, 2.0f / (t - b), 0.0f, -(t + b) / (t - b),
        0.0f, 0.0f, -2.0f / (f - n), -(f + n) / (f - n),
        0.0f, 0.0f, 0.0f, 1.0f);
}


cgVec3 cross(const cgVec3& v1, const cgVec3& v2){
    cgVec3 out = cgVec3();
    out.x = v1.y*v2.z-v1.z*v2.y;
    out.y = v1.z*v2.x-v1.x*v2.z;
    out.z = v1.x*v2.y-v1.y*v2.x;
    return out;
}

cgVec3 tripleProd(const cgVec3& v1,const cgVec3& v2,const cgVec3& v3){
    return cross(cross(v1,v2),v3);
}

cgMat3::cgMat3(float xx,float yx,float zx,
               float xy,float yy,float zy,
               float xz,float yz,float zz){

    data[0] = xx; //00
    data[1] = xy; //10
    data[2] = xz; //20
    data[3] = yx; 
    data[4] = yy; //11
    data[5] = yz;
    data[6] = zx;
    data[7] = zy;
    data[8] = zz;
}

cgMat3 cgMat3::operator*(const cgMat3& mat){
    cgMat3 output = cgMat3();
    float sum;
    for(int i = 0; i<3;i++){
        for(int j = 0;j<3;j++){
            sum = 0;
            for(int k = 0; k < 3;k++){
                sum = sum + at(i, k)*mat.at(k,j);
            }
            output.set(i, j, sum);
        }
    }
    return output;
}

void cgMat3::print(){
    for(int i = 0; i<3;i++){
        std::cout << "|";
        for(int j = 0;j<3;j++){
            std::cout << at(i, j) << " ";
        }
        std::cout << "|\n";
    }
}

cgVec3 cgMat3::operator*(const cgVec3& vec){
    return cgVec3(data[0]*vec.x+data[3]*vec.y+data[6]*vec.z, data[1]*vec.x+data[4]*vec.y+data[7]*vec.z, data[2]*vec.x+data[5]*vec.y+data[8]*vec.z);
}

float dot(cgVec3 v1,cgVec3 v2){
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

float cgMat3::determinant(){
    return (at(0,0)*at(1,1)*at(2,2)+at(0,1)*at(1,2)*at(2,0)+at(0,2)*at(1,0)*at(2,1))-(at(2,0)*at(1,1)*at(0,2)+at(2,1)*at(1,2)*at(0,0)+at(2,2)*at(1,0)*at(0,1));
}

cgMat3 cgMat3::inverse(){
    float det = determinant();
    cgMat3 inv = cgMat3();
    inv.set(0, 0, (at(1, 1) * at(2, 2) - at(2, 1) * at(1, 2)) * det);
    inv.set(0, 1, (at(0, 2) * at(2, 1) - at(0, 1) * at(2, 2)) * det);
    inv.set(0, 2, (at(0, 1) * at(1, 2) - at(0, 2) * at(1, 1)) * det);
    inv.set(1, 0, (at(1, 2) * at(2, 0) - at(1, 0) * at(2, 2)) * det);
    inv.set(1, 1, (at(0, 0) * at(2, 2) - at(0, 2) * at(2, 0)) * det);
    inv.set(1, 2, (at(1, 0) * at(0, 2) - at(0, 0) * at(1, 2)) * det);
    inv.set(2, 0, (at(1, 0) * at(2, 1) - at(2, 0) * at(1, 1)) * det);
    inv.set(2, 1, (at(2, 0) * at(0, 1) - at(0, 0) * at(2, 1)) * det);
    inv.set(2, 2, (at(0, 0) * at(1, 1) - at(1, 0) * at(0, 1)) * det);
    return inv;
}

quat axisAngle(cgVec3 axis, float theta){
    return quat(cos(theta/2),axis*sin(theta/2));
}

quat eulerAngles(float yaw, float pitch, float roll){
    // copied from wikipedia
    float cy = cos(roll * 0.5);
    float sy = sin(roll * 0.5);
    float cp = cos(pitch * 0.5);
    float sp = sin(pitch * 0.5);
    float cr = cos(yaw * 0.5);
    float sr = sin(yaw * 0.5);

    quat q;
    q.w = cr * cp * cy + sr * sp * sy;
    q.v.x = sr * cp * cy - cr * sp * sy;
    q.v.y = cr * sp * cy + sr * cp * sy;
    q.v.z = cr * cp * sy - sr * sp * cy;

    return q;
}

cgMat4 lookAt(cgVec3 eye, cgVec3 at, cgVec3 up)
{
    cgVec3 zaxis = (at - eye).normalized();    
    cgVec3 xaxis = cross(zaxis, up).normalized();
    cgVec3 yaxis = cross(xaxis, zaxis);

    zaxis = zaxis*(-1);
    
    cgMat4 viewMatrix = {xaxis.x, xaxis.y, xaxis.z, -dot(xaxis, eye),
                        yaxis.x, yaxis.y, yaxis.z, -dot(yaxis, eye),
                        zaxis.x, zaxis.y, zaxis.z, -dot(zaxis, eye),
                        0.0f, 0.0f, 0.0f, 1.0f};

    return viewMatrix;
}

cgVec3 lerp(cgVec3 p1, cgVec3 p2, float k){
    if(fabsf(p1.x - p2.x) < 0.001 && fabsf(p1.y - p2.y) < 0.001 && fabsf(p1.z - p2.z) < 0.001){
        return p1;
    }
    // return p1*k + p2*(1-k);
    return p1*(1-k) + p2*k;
}

float lerp(float x0, float x1, float t){
    if(fabsf(x0 - x1) < 0.001){
        return x0;
    }
    return (1-t)*x0 + t*x1;
}

cgVec2 lerp(cgVec2 x0, cgVec2 x1, float t){
    if(fabsf(x0.x - x1.x) < 0.001 && fabsf(x0.y - x1.y) < 0.001){
        return x0;
    }
    float xtx = lerp(x0.x,x1.x,t);
    float xty = lerp(x0.y,x1.y,t);
    return cgVec2(xtx,xty);
}

quat lerp(quat p1, quat p2, float k){
    if(k <= 0.001){
        return p1;
    }
    if(k >= 0.999){
        return p2;
    }
    //check if p1 and p2 are the same
    if(dot(p1,p2) > 0.999){
        return p1;
    }
    float a = 1.0 - k;
    float b = k;
    float d = dot(p1,p2);
    float c = fabsf(d);
    if(c < 0.999) {
        c = acosf(c);
        float sinC = sinf(c);
        if(sinC == 0.0f) {
            // Avoid division by zero - return p1 when angle is 0 or π
            return p1;
        }
        b = 1 / sinC;
        a = sinf(a * c) * b;
        b *= sinf(k * c);
        if(d < 0) b = -b;
    }
    quat ret;

    ret.w   = p1.w   * a + p2.w   * b;
    ret.v.x = p1.v.x * a + p2.v.x * b;
    ret.v.y = p1.v.y * a + p2.v.y * b;
    ret.v.z = p1.v.z * a + p2.v.z * b;
    return ret;

}

quat slerp(quat q1, quat q2, float t) {
    // modified version of glm::slerp
    quat z = q2;
    double cosTheta = dot(q1, q2);

    // If cosTheta < 0, the interpolation will take the long way around the sphere.
    // To fix this, one quat must be negated.
    if(cosTheta < 0)
    {
        z = q2*-1;
        cosTheta = -cosTheta;
    }

    // Perform a linear interpolation when cosTheta is close to 1 to avoid side effect of sin(angle) becoming a zero denominator
    if(cosTheta > (1.0 - std::numeric_limits<float>::epsilon()))
    {
        // Linear interpolation
        return quat(lerp(q1.w, z.w, t), lerp(q1.v, z.v, t));
    }
    else
    {
        // Essential Mathematics, page 467
        // Clamp cosTheta to [-1, 1] to handle floating point errors that could cause acos to return NaN
        cosTheta = std::max(-1.0, std::min(1.0, cosTheta));
        double angle = acos(cosTheta);
        double sinAngle = sin(angle);
        if(sinAngle == 0.0) {
            // Avoid division by zero - return q1 when angle is 0 or π
            return q1;
        }
        return (q1*sin((1 - t) * angle) + z*sin(t * angle)) * (1.0 / sinAngle);
    }
}

float dot(quat q1, quat q2){
    return q1.v.x * q2.v.x + q1.v.y * q2.v.y + q1.v.z * q2.v.z + q1.w * q2.w;
}

float fsign(float number){
    if(number < 0){
        return -1;
    } else if (number > 0){
        return 1;
    } else {
        return 0;
    }
}

quat getRotationBetweenVecs(cgVec3 start, cgVec3 end){
    float d = dot(start,end);
    cgVec3 xUnit = cgVec3(1,0,0);
    cgVec3 yUnit = cgVec3(0,1,0);

    if(d > 0.999){
        return quat(1,cgVec3(0,0,0));
    }
    if(d < -0.999){
        cgVec3 tmp = cross(xUnit, start);
        if (tmp.norm() < 0.000001)
            tmp = cross(yUnit, start);
        tmp = tmp.normalized();
        return axisAngle(tmp, M_PI);
    }

    cgVec3 a = cross(start,end);
    quat out = quat(1+d,a);
    out.normalize();
    return out;
}

quat alignCoordinateSystems(
    cgVec3 initialX,
    cgVec3 initialY,
    cgVec3 finalX,
    cgVec3 finalY
){
    
    cgVec3 axis0 = cross(initialX.normalized(),finalX.normalized());
    quat rot0 = eulerAngles(0,0,0);
    float sinAngle0 = axis0.norm();
    if(sinAngle0 < 0.01){
        //axis0 will be weird
        axis0 = initialX.getOrthogonalVector();
    }

    axis0 = axis0.normalized();
    float ix_len = initialX.norm();
    float fx_len = finalX.norm();
    float cosAngle0 = dot(initialX, finalX)/(ix_len*fx_len);
    float angle0 = atan2(sinAngle0,cosAngle0);
    rot0 = axisAngle(axis0, angle0);

    cgVec3 initialY_adj = rot0.rotate(initialY);
    
    cgVec3 axis1 = finalX.normalized();//cross(initialY_adj.normalized(),finalY.normalized());
    quat rot1 = eulerAngles(0,0,0);
    float iy_len = initialY_adj.norm();
    float fy_len = finalY.norm();
    float cosAngle1 = dot(initialY_adj, finalY)/(iy_len*fy_len);
    // Clamp to [-1, 1] to handle floating point errors that could cause acos to return NaN
    cosAngle1 = std::max(-1.0f, std::min(1.0f, cosAngle1));
    float angle1 = acos(cosAngle1);//asin(sinAngle1);

    rot1 = axisAngle(axis1, angle1);

    // if(dot((rot1*rot0).rotate(initialY),finalY) < 0.999 ){
    // } else {
    //     // std::cout << "nolock: ";axis0.print();
    // }
    return (rot1*rot0);
}

void extractTRS(cgVec3& translation, quat& rotation, cgVec3& scale, cgMat4 matrix){
    translation.x = matrix.at(0,3);
    translation.y = matrix.at(1,3);
    translation.z = matrix.at(2,3);

    matrix.set(0,3,0);
    matrix.set(1,3,0);
    matrix.set(2,3,0);

    scale.x = cgVec3(matrix.at(0,0),matrix.at(1,0),matrix.at(2,0)).norm();
    scale.y = cgVec3(matrix.at(0,1),matrix.at(1,1),matrix.at(2,1)).norm();
    scale.z = cgVec3(matrix.at(0,2),matrix.at(1,2),matrix.at(2,2)).norm();

    matrix.set(0,0,matrix.at(0,0)/scale.x);
    matrix.set(1,0,matrix.at(1,0)/scale.x);
    matrix.set(2,0,matrix.at(2,0)/scale.x);

    matrix.set(0,1,matrix.at(0,1)/scale.y);
    matrix.set(1,1,matrix.at(1,1)/scale.y);
    matrix.set(2,1,matrix.at(2,1)/scale.y);

    matrix.set(0,2,matrix.at(0,2)/scale.z);
    matrix.set(1,2,matrix.at(1,2)/scale.z);
    matrix.set(2,2,matrix.at(2,2)/scale.z);

    rotation = quat(matrix);
    
}

