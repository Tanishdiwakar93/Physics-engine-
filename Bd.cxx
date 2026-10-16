#include <SDL2/SDL.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

// ==========================================
// 1. ADVANCED 3D MATH ENGINE
// ==========================================

struct Vec3 {
    float x = 0, y = 0, z = 0;

    Vec3 operator+(const Vec3& v) const { return {x + v.x, y + v.y, z + v.z}; }
    Vec3 operator-(const Vec3& v) const { return {x - v.x, y - v.y, z - v.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    Vec3& operator+=(const Vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    Vec3& operator-=(const Vec3& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
    
    float dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    Vec3 cross(const Vec3& v) const {
        return { y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x };
    }
    float length() const { return std::sqrt(x * x + y * y + z * z); }
    Vec3 normalized() const { float l = length(); return l > 0 ? *this * (1.0f / l) : Vec3{}; }
};

struct Mat3 {
    float m[9] = {1,0,0, 0,1,0, 0,0,1};

    static Mat3 Identity() { return Mat3(); }
    
    Vec3 operator*(const Vec3& v) const {
        return {
            m[0]*v.x + m[1]*v.y + m[2]*v.z,
            m[3]*v.x + m[4]*v.y + m[5]*v.z,
            m[6]*v.x + m[7]*v.y + m[8]*v.z
        };
    }

    Mat3 operator*(const Mat3& o) const {
        Mat3 r;
        for(int i=0; i<3; ++i) {
            for(int j=0; j<3; ++j) {
                r.m[i*3 + j] = m[i*3 + 0]*o.m[0*3 + j] +
                              m[i*3 + 1]*o.m[1*3 + j] +
                              m[i*3 + 2]*o.m[2*3 + j];
            }
        }
        return r;
    }

    Mat3 transposed() const {
        Mat3 r;
        r.m[0]=m[0]; r.m[1]=m[3]; r.m[2]=m[6];
        r.m[3]=m[1]; r.m[4]=m[4]; r.m[5]=m[7];
        r.m[6]=m[2]; r.m[7]=m[5]; r.m[8]=m[8];
        return r;
    }

    Vec3 getAxis(int i) const {
        return { m[i], m[i+3], m[i+6] };
    }
};

struct Quaternion {
    float w = 1.0f, x = 0.0f, y = 0.0f, z = 0.0f;

    Quaternion normalized() const {
        float len = std::sqrt(w*w + x*x + y*y + z*z);
        if (len == 0) return {1, 0, 0, 0};
        return {w/len, x/len, y/len, z/len};
    }

    Quaternion operator*(const Quaternion& q) const {
        return {
            w*q.w - x*q.x - y*q.y - z*q.z,
            w*q.x + x*q.w + y*q.z - z*q.y,
            w*q.y - x*q.z + y*q.w + z*q.x,
            w*q.z + x*q.y - y*q.x + z*q.w
        };
    }

    Quaternion operator*(float s) const {
        return { w * s, x * s, y * s, z * s };
    }

    Mat3 toMatrix() const {
        Mat3 m;
        m.m[0] = 1 - 2*y*y - 2*z*z; m.m[1] = 2*x*y - 2*w*z;   m.m[2] = 2*x*z + 2*w*y;
        m.m[3] = 2*x*y + 2*w*z;   m.m[4] = 1 - 2*x*x - 2*z*z; m.m[5] = 2*y*z - 2*w*x;
        m.m[6] = 2*x*z - 2*w*y;   m.m[7] = 2*y*z + 2*w*x;   m.m[8] = 1 - 2*x*x - 2*y*y;
        return m;
    }
};

// ==========================================
// 2. RIGID BODY DYNAMICS
// ==========================================

struct RigidBody3D {
    Vec3 position;
    Vec3 linearVelocity;
    Vec3 angularVelocity;
    
    Quaternion orientation;
    Vec3 halfSize = {0.5f, 0.5f, 0.5f};

    float mass = 1.0f;
    float invMass = 1.0f;
    
    Mat3 invInertiaLocal;
    Mat3 invInertiaWorld;

    Vec3 forceAccum;
    Vec3 torqueAccum;

    float restitution = 0.3f;

    RigidBody3D(Vec3 pos, Vec3 size, float m) : position(pos), halfSize(size), mass(m) {
        if (m > 0.0f) {
            invMass = 1.0f / m;
            float x2 = (size.x * 2) * (size.x * 2);
            float y2 = (size.y * 2) * (size.y * 2);
            float z2 = (size.z * 2) * (size.z * 2);
            
            Mat3 inertiaLocal;
            inertiaLocal.m[0] = (1.0f / 12.0f) * m * (y2 + z2);
            inertiaLocal.m[4] = (1.0f / 12.0f) * m * (x2 + z2);
            inertiaLocal.m[8] = (1.0f / 12.0f) * m * (x2 + y2);
            
            invInertiaLocal.m[0] = 1.0f / inertiaLocal.m[0];
            invInertiaLocal.m[4] = 1.0f / inertiaLocal.m[4];
            invInertiaLocal.m[8] = 1.0f / inertiaLocal.m[8];
        } else {
            invMass = 0.0f;
            invInertiaLocal = Mat3{};
            for(int i=0; i<9; ++i) invInertiaLocal.m[i] = 0;
        }
    }

    void updateInertiaWorld() {
        if (invMass == 0.0f) return;
        Mat3 R = orientation.toMatrix();
        invInertiaWorld = R * invInertiaLocal * R.transposed();
    }

    void applyForceAtPoint(const Vec3& force, const Vec3& point) {
        forceAccum += force;
        torqueAccum += (point - position).cross(force);
    }

    void integrate(float dt) {
        if (invMass == 0.0f) return;

        Vec3 linAccel = forceAccum * invMass;
        linearVelocity += linAccel * dt;
        position += linearVelocity * dt;

        Vec3 angAccel = invInertiaWorld * torqueAccum;
        angularVelocity += angAccel * dt;

        Quaternion wQuat = {0, angularVelocity.x, angularVelocity.y, angularVelocity.z};
        Quaternion dq = (wQuat * orientation) * (0.5f * dt);
        orientation.w += dq.w; orientation.x += dq.x; 
        orientation.y += dq.y; orientation.z += dq.z;
        orientation = orientation.normalized();

        updateInertiaWorld();

        forceAccum = {0, 0, 0};
        torqueAccum = {0, 0, 0};
    }

    std::vector<Vec3> getVertices() const {
        Mat3 R = orientation.toMatrix();
        std::vector<Vec3> verts;
        Vec3 signs[8] = {
            {-1,-1,-1}, { 1,-1,-1}, { 1, 1,-1}, {-1, 1,-1},
            {-1,-1, 1}, { 1,-1, 1}, { 1, 1, 1}, {-1, 1, 1}
        };
        for (int i = 0; i < 8; ++i) {
            Vec3 local = {signs[i].x * halfSize.x, signs[i].y * halfSize.y, signs[i].z * halfSize.z};
            verts.push_back(position + (R * local));
        }
        return verts;
    }
};

// ==========================================
// 3. CONTACT RESOLUTION & BOX-BOX SAT SOLVER
// ==========================================

struct Contact {
    RigidBody3D* bodyA;
    RigidBody3D* bodyB;
    Vec3 point;
    Vec3 normal;
    float penetration;
};

void resolveContact(Contact& c) {
    RigidBody3D* a = c.bodyA;
    RigidBody3D* b = c.bodyB;

    Vec3 rA = c.point - a->position;
    Vec3 rB = c.point - b->position;

    Vec3 vA = a->linearVelocity + a->angularVelocity.cross(rA);
    Vec3 vB = b->linearVelocity + b->angularVelocity.cross(rB);
    Vec3 relVel = vA - vB;

    float velAlongNormal = relVel.dot(c.normal);
    if (velAlongNormal > 0) return;

    float e = std::min(a->restitution, b->restitution);

    float rAcn = rA.cross(c.normal).dot(a->invInertiaWorld * rA.cross(c.normal));
    float rBcn = rB.cross(c.normal).dot(b->invInertiaWorld * rB.cross(c.normal));

    float invMassSum = a->invMass + b->invMass + rAcn + rBcn;
    if (invMassSum == 0.0f) return;

    float j = -(1.0f + e) * velAlongNormal / invMassSum;
    Vec3 impulse = c.normal * j;

    if (a->invMass > 0) {
        a->linearVelocity += impulse * a->invMass;
        a->angularVelocity += a->invInertiaWorld * rA.cross(impulse);
    }
    if (b->invMass > 0) {
        b->linearVelocity -= impulse * b->invMass;
        b->angularVelocity -= b->invInertiaWorld * rB.cross(impulse);
    }

    const float percent = 0.4f;
    const float slop = 0.01f;
    Vec3 correction = c.normal * (std::max(c.penetration - slop, 0.0f) / (a->invMass + b->invMass) * percent);
    if (a->invMass > 0) a->position += correction * a->invMass;
    if (b->invMass > 0) b->position -= correction * b->invMass;
}

void projectBoxOnAxis(const RigidBody3D& box, const Vec3& axis, float& minP, float& maxP) {
    auto verts = box.getVertices();
    minP = maxP = verts[0].dot(axis);
    for (size_t i = 1; i < verts.size(); ++i) {
        float p = verts[i].dot(axis);
        if (p < minP) minP = p;
        if (p > maxP) maxP = p;
    }
}

bool checkBoxBoxCollision(RigidBody3D& boxA, RigidBody3D& boxB, Contact& contact) {
    Mat3 RA = boxA.orientation.toMatrix();
    Mat3 RB = boxB.orientation.toMatrix();

    Vec3 axes[15];
    for (int i = 0; i < 3; ++i) axes[i] = RA.getAxis(i);
    for (int i = 0; i < 3; ++i) axes[i + 3] = RB.getAxis(i);

    int idx = 6;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            axes[idx++] = axes[i].cross(axes[j]);
        }
    }

    float minOverlap = std::numeric_limits<float>::max();
    Vec3 bestAxis;

    for (int i = 0; i < 15; ++i) {
        Vec3 axis = axes[i];
        if (axis.length() < 0.001f) continue;
        axis = axis.normalized();

        float minA, maxA, minB, maxB;
        projectBoxOnAxis(boxA, axis, minA, maxA);
        projectBoxOnAxis(boxB, axis, minB, maxB);

        float overlap = std::min(maxA, maxB) - std::max(minA, minB);
        if (overlap <= 0.0f) return false;

        if (overlap < minOverlap) {
            minOverlap = overlap;
            bestAxis = axis;
        }
    }

    if ((boxA.position - boxB.position).dot(bestAxis) < 0) {
        bestAxis = bestAxis * -1.0f;
    }

    contact.bodyA = &boxA;
    contact.bodyB = &boxB;
    contact.normal = bestAxis;
    contact.penetration = minOverlap;
    contact.point = (boxA.position + boxB.position) * 0.5f;

    return true;
}

void checkFloorCollisions(RigidBody3D& body, float floorY) {
    auto verts = body.getVertices();
    for (const auto& v : verts) {
        if (v.y < floorY) {
            Contact c;
            c.bodyA = &body;
            static RigidBody3D floor({0, floorY, 0}, {100, 0.1f, 100}, 0.0f);
            c.bodyB = &floor;
            c.point = v;
            c.normal = {0, 1.0f, 0};
            c.penetration = floorY - v.y;
            resolveContact(c);
        }
    }
}

// ==========================================
// 4. RENDERER & MAIN LOOP (CENTER ALIGNED)
// ==========================================

struct SDL_Point2D { int x, y; };

SDL_Point2D project3DTo2D(Vec3 p, int screenW, int screenH) {
    float fov = 400.0f;
    // Camera is positioned back and slightly up to center the floor/scene
    Vec3 camPos = {0.0f, 1.5f, -10.0f};
    
    Vec3 localP = p - camPos;
    if (localP.z <= 0.1f) localP.z = 0.1f;

    int x = (int)((localP.x * fov) / localP.z + screenW / 2);
    int y = (int)((-localP.y * fov) / localP.z + screenH / 2);
    return {x, y};
}

void drawWireframeBox(SDL_Renderer* renderer, const RigidBody3D& body, int screenW, int screenH) {
    auto verts = body.getVertices();
    std::vector<SDL_Point2D> pts(8);
    for (int i = 0; i < 8; ++i) {
        pts[i] = project3DTo2D(verts[i], screenW, screenH);
    }

    int edges[12][2] = {
        {0,1}, {1,2}, {2,3}, {3,0},
        {4,5}, {5,6}, {6,7}, {7,4},
        {0,4}, {1,5}, {2,6}, {3,7}
    };

    SDL_SetRenderDrawColor(renderer, 0, 255, 120, 255); // Green Box
    for (int i = 0; i < 12; ++i) {
        SDL_RenderDrawLine(renderer, 
            pts[edges[i][0]].x, pts[edges[i][0]].y,
            pts[edges[i][1]].x, pts[edges[i][1]].y
        );
    }
}

void drawFloorGrid(SDL_Renderer* renderer, float floorY, int screenW, int screenH) {
    SDL_SetRenderDrawColor(renderer, 60, 60, 80, 255);
    for (float x = -4.0f; x <= 4.0f; x += 1.0f) {
        auto p1 = project3DTo2D({x, floorY, -4.0f}, screenW, screenH);
        auto p2 = project3DTo2D({x, floorY, 4.0f}, screenW, screenH);
        SDL_RenderDrawLine(renderer, p1.x, p1.y, p2.x, p2.y);
    }
    for (float z = -4.0f; z <= 4.0f; z += 1.0f) {
        auto p1 = project3DTo2D({-4.0f, floorY, z}, screenW, screenH);
        auto p2 = project3DTo2D({4.0f, floorY, z}, screenW, screenH);
        SDL_RenderDrawLine(renderer, p1.x, p1.y, p2.x, p2.y);
    }
}

int main(int argc, char* argv[]) {
    SDL_Init(SDL_INIT_VIDEO);

    int screenW = 720, screenH = 1280;
    SDL_Window* window = SDL_CreateWindow("3D Physics Engine", 
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 
        screenW, screenH, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    std::vector<RigidBody3D> world;

    float floorLevel = -2.0f; // Floor level

    // Initial resting box right on top of the floor
    RigidBody3D box1({0.0f, floorLevel + 0.5f, 0.0f}, {0.5f, 0.5f, 0.5f}, 2.0f);
    world.push_back(box1);

    Vec3 gravity{0.0f, -9.81f, 0.0f};
    float dt = 0.016f;
    bool running = true;
    SDL_Event e;

    while (running) {
        // Fetch current display size dynamically for exact center calculation
        SDL_GetRendererOutputSize(renderer, &screenW, &screenH);

        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = false;
            if (e.type == SDL_MOUSEBUTTONDOWN) {
                float offsetX = ((rand() % 100) / 100.0f) - 0.5f;
                // Spawn new box dropping from above
                RigidBody3D newBox({offsetX, 3.0f, 0.0f}, {0.5f, 0.5f, 0.5f}, 1.5f);
                newBox.angularVelocity = {2.0f, 1.0f, 1.5f};
                world.push_back(newBox);
            }
        }

        // --- Physics Step ---
        for (auto& body : world) {
            body.applyForceAtPoint(gravity * body.mass, body.position);
            body.integrate(dt);
            checkFloorCollisions(body, floorLevel);
        }

        // --- Box vs Box Collisions ---
        for (size_t i = 0; i < world.size(); ++i) {
            for (size_t j = i + 1; j < world.size(); ++j) {
                Contact c;
                if (checkBoxBoxCollision(world[i], world[j], c)) {
                    resolveContact(c);
                }
            }
        }

        // --- Render Step ---
        SDL_SetRenderDrawColor(renderer, 20, 20, 25, 255);
        SDL_RenderClear(renderer);

        drawFloorGrid(renderer, floorLevel, screenW, screenH);

        for (const auto& body : world) {
            drawWireframeBox(renderer, body, screenW, screenH);
        }

        SDL_RenderPresent(renderer);
        SDL_Delay(16);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
