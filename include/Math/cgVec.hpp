#ifndef CGVEC
#define CGVEC
#include <stdio.h>
#include "math.h"
#include <iostream>
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

#ifndef M_PI_2
#define M_PI_2 (M_PI / 2.0) 
#endif

#ifndef M_PI_4
#define M_PI_4 (M_PI / 4.0)
#endif

struct cgMat4;

struct cgVec2Int{
    int x;
    int y;
    cgVec2Int(){};
    cgVec2Int(int x,int y){this->x=x;this->y=y;}
    cgVec2Int(const cgVec2Int& vec){this->x=vec.x;this->y=vec.y;}
    void print(){std::cout << "( "<<x<<","<<y<<")\n";}
    bool operator==(const cgVec2Int& vector){return (this->x==vector.x && this->y==vector.y);};
    cgVec2Int operator-(const cgVec2Int& vector){return cgVec2Int(x-vector.x,y-vector.y);};
    cgVec2Int operator+(const cgVec2Int& vector){return cgVec2Int(vector.x+x,vector.y+y);};
};

struct cgVec2{
    float x;
    float y;
    cgVec2(){};
    cgVec2(float x,float y);
    void print(){std::cout <<"("<< x <<", " << y << " )\n";}
    cgVec2(const cgVec2& vector); //Copy constructor
    cgVec2 operator+(const cgVec2& vector);
    cgVec2 operator-(const cgVec2& vector);
    cgVec2 operator*(const float& scalar);
    float innerProd(const cgVec2& vector);
    float mag(){return sqrt(x*x+y*y);}
    cgVec2 normalized();
    cgVec2 rotatedBy(float radians);
    void rotateBy(float radians);
};

float dot(cgVec2 v1,cgVec2 v2);

struct cgMat2{
    cgMat2(float r1c1, float r1c2,
           float r2c1, float r2c2);
    float at(int row, int col);
    cgVec2 operator*(const cgVec2& vector);
private:
    float data[4];
};

class cg2DFloatArray{
public:
    cg2DFloatArray(int numRows, int numCols);
    ~cg2DFloatArray();
    float at(int row, int col);
    void set(int row, int col, float to);
    float max();
    float min();
public:
    int numRows;
    int numCols;
private:
    float**data;
};

struct cgVec3{
    float x;
    float y;
    float z;
    cgVec3(){this->x = 0;this->y = 0;this->z = 0;};
    cgVec3(float x,float y, float z){this->x = x;this->y = y;this->z = z;};
    cgVec3(const cgVec3& vec1){x = vec1.x;y = vec1.y;z = vec1.z;}
    float at(int index){return *(((float*)this)+index);};
    void set(int index, float to){ *(((float*)this)+index) = to;};
    void print(){std::cout <<"("<< x <<", " << y << ", " << z << ", " << " )\n";}
    void add(cgVec3 vec){x = x+vec.x;y = y+vec.y;z = z+vec.z;}
    cgVec3 operator+(const cgVec3& vector) {return cgVec3(vector.x+x,vector.y+y,vector.z+z);}
    cgVec3 operator-(const cgVec3& vector) const {
        return cgVec3(x - vector.x, y - vector.y, z - vector.z);
    }
    //cgVec3 operator-(const cgVec3& vector) {return cgVec3(vector.x-x,vector.y-y,vector.z-z);}
    cgVec3 operator*(const float scalar) {return cgVec3(x*scalar,y*scalar,z*scalar);}
    cgVec3 operator/(const float scalar) {return cgVec3(x/scalar,y/scalar,z/scalar);}
    void operator=(const cgVec3& v){x=v.x;y=v.y;z=v.z;}

    //element wise multiplication
    cgVec3 operator*(const cgVec3& v){return {v.x*x,v.y*y,v.z*z}; }
    
    cgVec3 scale(float scalar){return cgVec3(x*scalar,y*scalar,z*scalar);}
    cgVec3 crossProd(const cgVec3& vector);
    float dot(const cgVec3& v){return v.x*x + v.y*y + v.z*z;}
    cgVec3 matMult4(const cgMat4& mat); //trasnforms 3d vector in the same way as shader
    void normalize();
    cgVec3 normalized(){float n = norm(); return cgVec3(x/n,y/n,z/n);}
    float norm(); 

    /**
     * @brief Generates a set 2 of orthogonal vectors in the plane orthogonal to this vector
     * the two resulting vectors are orthogonal to each other
     */
    void getOrthogonalVectors(cgVec3& target1, cgVec3& target2);

    //gets a vector orthogonal to this one
    cgVec3 getOrthogonalVector(); 

    static cgVec3 ExtractPosition(const cgMat4& matrix); 
    static cgVec3 ExtractDirection(const cgMat4& matrix);

    float length() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    float lengthSq() const {
        return x * x + y * y + z * z;
    }

    bool isNaN(){return isnan(x) || isnan(y) || isnan(z);}
};

struct cgMat3{
    float data[9];
    cgMat3(float xx,float yx,float zx,
           float xy,float yy,float zy,
           float xz,float yz,float zz);
    cgMat3(){for(int i = 0; i < 9; i++){data[i] = 0;}}
    cgMat3(const cgMat3& mat){for(int i = 0; i < 9; i++){data[i] = mat.data[i];}}
    float at(int r,int c) const {return data[3*c + r];}
    void set(int r,int c, float to){data[3*c + r] = to;}
    void zeros(){for(int i = 0; i < 9; i++){data[i] = 0;}} 
    void toIdentity(){this->zeros(); this->set(0,0,1);this->set(1,1,1);this->set(2,2,1);}
    cgMat3 operator*(const cgMat3& mat);
    cgVec3 operator*(const cgVec3& vec);
    void print();
    float determinant();
    cgMat3 inverse();
    cgMat3 transposed(){return cgMat3(at(0,0),at(1,0),at(2,0),at(0,1),at(1,1),at(2,1),at(0,2),at(1,2),at(2,2) ); }
    bool isNaN(){return isnan(data[0]) || isnan(data[1]) || isnan(data[2]) || isnan(data[3]) || isnan(data[4]) || isnan(data[5]) || isnan(data[6]) || isnan(data[7]) || isnan(data[8]);}
};

struct cgVec4{
    float x;
    float y;
    float z;
    float w;
    cgVec4(float x,float y, float z,float w){this->x = x;this->y = y;this->z = z;this->w = w;};
    cgVec4(){this->x = 0;this->y = 0;this->z = 0;this->w = 0;}
    cgVec4(const cgVec4& v){this->x = v.x;this->y = v.y;this->z = v.z;this->w = v.w;}
    cgVec4(cgVec3 vec){this->x = vec.x;this->y = vec.y;this->z = vec.z;this->w = 1.0;}//sets the last element to 1
    cgVec3 toVec3(){return cgVec3(x,y,z);}
    bool isNaN(){return isnan(x) || isnan(y) || isnan(z) || isnan(w);}
};

struct cgMat4{
    float data[16];
    cgMat4(float xx,float yx,float zx,float wx,
           float xy,float yy,float zy,float wy,
           float xz,float yz,float zz,float wz,
           float xw,float yw,float zw,float ww);
    cgMat4(){for(int i = 0; i < 16; i++){data[i] = 0;}}
    cgMat4(const cgMat4& mat){for(int i = 0; i < 16; i++){data[i] = mat.data[i];}}
    inline float at(int r,int c) const {return data[4*c + r];}
    inline float* colPtr(int c) {return data + 4*c;}
    inline const float* colPtr_c(int c) const {return data + 4*c;}
    // inline const float* colPtr_c(int c) const {return data + 4*c;}
    inline void set(int r,int c, float to){data[4*c + r] = to;}
    void zeros(){for(int i = 0; i < 16; i++){data[i] = 0;}} 
    void toIdentity(){this->zeros(); this->set(0,0,1);this->set(1,1,1);this->set(2,2,1);this->set(3,3,1);}
    cgMat4 operator*(const cgMat4& mat) const;
    cgVec4 operator*(const cgVec4& vec) const;
    cgVec3 operator*(const cgVec3& vec) const;
    void print();
    void removeTranslation();//sets bottom and right side to 0 (removes translation)
    cgMat3 toMat3(){return cgMat3(at(0,0),at(0,1),at(0,2),at(1,0),at(1,1),at(1,2),at(2,0),at(2,1),at(2,2) ); };

    float determinant();
    cgMat4 inverse();
    cgMat4 transposed(){return cgMat4(at(0,0),at(1,0),at(2,0),at(3,0),at(0,1),at(1,1),at(2,1),at(3,1),at(0,2),at(1,2),at(2,2),at(3,2),at(0,3),at(1,3),at(2,3),at(3,3) ); }
    bool isNaN(){return isnan(data[0]) || isnan(data[1]) || isnan(data[2]) || isnan(data[3]) || isnan(data[4]) || isnan(data[5]) || isnan(data[6]) || isnan(data[7]) || isnan(data[8]) || isnan(data[9]) || isnan(data[10]) || isnan(data[11]) || isnan(data[12]) || isnan(data[13]) || isnan(data[14]) || isnan(data[15]);}
};

cgMat4 rot(cgVec3 axis, float theta);

struct quat{ //quaternion 
    float w;
    cgVec3 v;
    quat(){this->v = cgVec3(0,0,0); w=1;}
    quat(float w, cgVec3 v){this->w=w;this->v=v;}
    quat(const quat& q){this->w=q.w;this->v=q.v;}

    /**
     * @brief Construct a new quat object from a rotation matrix
     * 
     * @param rotationMatrix 
     */
    quat(cgMat4 rm);

    void normalize(){float n = getNorm(); v = v*(1/n); w=w/n;}
    float getNorm(){return sqrt(v.x*v.x+v.y*v.y+v.z*v.z+w*w);}
    quat operator+(const quat& q){return quat(w+q.w,v+q.v);}
    quat operator*(quat q){return quat(w*q.w-v.dot(q.v),v.crossProd(q.v)+((q.v).scale(w)+v.scale(q.w)) );}
    quat operator*(float scalar){return quat(w*scalar,v*scalar);}
    quat operator*(const cgVec3& vec){quat temp = quat(0,vec); return (*this)*temp;}
    quat conjugate(){return quat(w,this->v*-1);}
    bool operator==(const quat& q){return ((w==q.w) && (v.x==q.v.x) && (v.y==q.v.y) && (v.z==q.v.z));}
    
    quat inv(){
        float norm = getNorm();
        return this->conjugate()*(1/(norm*norm));
    }
    cgMat4 getRotMatrix();
    
    cgVec3 rotate(cgVec3 vector){
        quat p = quat(0,vector);
        quat qi = inv();
        quat res = (*this)*p*qi;
        return res.v;
    }

    void print(){std::cout << "("<<v.x<<", "<<v.y<<", "<<v.z<<", "<<w<<")\n";}
    void printEuler(){float yaw, pitch, roll; getEulerAngles(yaw, pitch, roll); std::cout << "yaw: "<<yaw<<" pitch: "<<pitch <<" roll: "<<roll<<"\n";}

    void getEulerAngles(float& yaw, float& pitch, float& roll);

    bool isNaN(){return isnan(v.x) || isnan(v.y) || isnan(v.z) || isnan(w);}
};

cgMat4* translation(float Tx, float Ty, float Tz);
cgMat4 trans(float Tx, float Ty, float Tz);
void translation(float Tx, float Ty, float Tz, cgMat4& mat);
cgMat4* scaling(float Sx, float Sy, float Sz);
cgMat4 scale(float Sx, float Sy, float Sz);

/**
 * @brief Orthographic projection from points describing the orthographic frustum
 * 
 * @param left 
 * @param right 
 * @param bottom 
 * @param top 
 * @param zNear 
 * @param zFar 
 * @return cgMat4 
 */
cgMat4 ortho(const float left, 
             const float right, 
             const float bottom, 
             const float top, 
             const float zNear, 
             const float zFar);

/**
 * @brief View matrix to look at the pt Center fron point eye with oriented with up pointing up
 * 
 * @param eye 
 * @param center 
 * @param up 
 * @return cgMat4 
 */
cgMat4 lookAt(cgVec3 eye, cgVec3 center, cgVec3 up);

/**
 * @brief A different (hopefully the same) implementation of lookAt
 * 
 * @param eye 
 * @param center 
 * @param up 
 * @return cgMat4 
 */
cgMat4 lookAt_glm(cgVec3 eye, cgVec3 center, cgVec3 up);

quat axisAngle(cgVec3 axis, float theta);
quat eulerAngles(float yaw, float pitch, float roll);

cgMat4* rotation(cgVec3 axis, float theta);
cgMat4 rot(float a, float b, float g); //yaw pitch roll (alpha beta gamma)

/**
 * @brief projection mat to look at "at" from cam position "eye" and orientation given by 
 * up vector
 * 
 * @param eye 
 * @param at 
 * @param up 
 * @return cgMat4 
 */
cgMat4 lookAt(cgVec3 eye, cgVec3 at, cgVec3 up);

void rotation(cgVec3 axis, float theta, cgMat4& mat);
cgMat4* projectionMatrix(float n, float r, float t, float f);
cgMat4* orthoProjection(float n, float r, float t, float f);
cgMat4* orthoProjectionWithBounds(float l, float r, float b, float t, float n, float f);
cgMat4* projectionMatrixSimple(float aspect, float fov, float n, float f);
cgVec3 cross(const cgVec3& v1, const cgVec3& v2);
cgVec3 tripleProd(const cgVec3& v1,const cgVec3& v2,const cgVec3& v3);
float dot(cgVec3 v1,cgVec3 v2);
float dot(quat q1, quat q2);

//interpolation

//linear interpolation between 2 points
cgVec3 lerp(cgVec3 p1, cgVec3 p2, float k);

//1d interpolation
float lerp(float x0, float x1, float t);

//2d interpolation between 2 points
cgVec2 lerp(cgVec2 x0, cgVec2 x1, float t);

//linear interpolation between 2 points
quat lerp(quat p1, quat p2, float k);

// spherical linear interpolation between quats
quat slerp(quat q1, quat q2, float t);

// /**
//  * @brief Quaternion to transform between coordinate system 1 (unit vectors x1,y1,z1) to coordinate system 2 (unit vectors x2,y2,z2)
//  * 
//  * @param x1 
//  * @param y1 
//  * @param z1 
//  * @param x2 
//  * @param y2 
//  * @param z2 
//  * @return quat 
//  */
// quat quatBetweenCoordinateSystems(cgVec3 x1, cgVec3 y1, cgVec3 z1, cgVec3 x2, cgVec3 y2, cgVec3 z2);

float fsign(float number);

/**
 * @brief Get a Rotation to align the start vector to the end vector
 * 
 * @param start 
 * @param end 
 * @return quat 
 */
quat getRotationBetweenVecs(cgVec3 start, cgVec3 end);

/**
 * @brief Return a quaternion that rotates the initial axes to the final axes, this is slightly broken so beware
 * 
 * @param initialX 
 * @param initialY 
 * @param initialZ 
 * @param finalX 
 * @param finalY 
 * @param finalZ 
 * @return quat 
 */
quat alignCoordinateSystems(
    cgVec3 initialX,
    cgVec3 initialY,
    cgVec3 finalX,
    cgVec3 finalY
);

/**
 * @brief Convert the transform matrix to a translation, rotation and scale
 * 
 * @param translation 
 * @param rotation 
 * @param scale 
 * @param matrix 
 */
void extractTRS(cgVec3& translation, quat& rotation, cgVec3& scale, cgMat4 matrix);

#endif /* CGVEC */
