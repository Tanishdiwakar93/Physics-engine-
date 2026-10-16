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
// 2. RIGID BODY DYNAMICS WITH DAMPING
// ==========================================

enum ShapeType { SHAPE_BOX, SHAPE_SPHERE };

struct RigidBody3D {
    ShapeType shape = SHAPE_BOX;
    Vec3 position;
    Vec3 linearVelocity;
    Vec3 angularVelocity;
    
    Quaternion orientation;
    Vec3 halfSize = {0.5f, 0.5f, 0.5f};
    float radius = 0.5f;

    float mass = 1.0f;
    float invMass = 1.0f;
    
    Mat3 invInertiaLocal;
    Mat3 invInertiaWorld;

    Vec3 forceAccum;
    Vec3 torqueAccum;

    float restitution = 0.3f;
    float friction = 0.5f;

    RigidBody3D(Vec3 pos, Vec3 size, float m, ShapeType st = SHAPE_BOX) 
        : position(pos), halfSize(size), radius(size.x), mass(m), shape(st) {
        if (m > 0.0f) {
            invMass = 1.0f / m;
            Mat3 inertiaLocal;

            if (shape == SHAPE_BOX) {
                float x2 = (size.x * 2) * (size.x * 2);
                float y2 = (size.y * 2) * (size.y * 2);
                float z2 = (size.z * 2) * (size.z * 2);
                inertiaLocal.m[0] = (1.0f / 12.0f) * m * (y2 + z2);
                inertiaLocal.m[4] = (1.0f / 12.0f) * m * (x2 + z2);
                inertiaLocal.m[8] = (1.0f / 12.0f) * m * (x2 + y2);
            } else {
                float I = (2.0f / 5.0f) * m * radius * radius;
                inertiaLocal.m[0] = I; inertiaLocal.m[4] = I; inertiaLocal.m[8] = I;
            }
            
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

    void applyImpulse(const Vec3& impulse, const Vec3& point) {
        if (invMass == 0.0f) return;
        linearVelocity += impulse * invMass;
        angularVelocity += invInertiaWorld * (point - position).cross(impulse);
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

        // Damping for stacking stability
        linearVelocity = linearVelocity * 0.98f;
        angularVelocity = angularVelocity * 0.95f;

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
// 3. CONTACT RESOLUTION & STACKING SOLVER
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

    // Friction
    Vec3 tangent = relVel - (c.normal * velAlongNormal);
    if (tangent.length() > 0.001f) {
        tangent = tangent.normalized();
        float jt = -relVel.dot(tangent) / invMassSum;
        float mu = std::sqrt(a->friction * b->friction);
        
        jt = std::clamp(jt, -j * mu, j * mu);
        Vec3 frictionImpulse = tangent * jt;

        if (a->invMass > 0) {
            a->linearVelocity += frictionImpulse * a->invMass;
            a->angularVelocity += a->invInertiaWorld * rA.cross(frictionImpulse);
        }
        if (b->invMass > 0) {
            b->linearVelocity -= frictionImpulse * b->invMass;
            b->angularVelocity -= b->invInertiaWorld * rB.cross(frictionImpulse);
        }
    }

    // Positional Projection to Prevent Sinking
    const float percent = 0.5f;
    const float slop = 0.01f;
    Vec3 correction = c.normal * (std::max(c.penetration - slop, 0.0f) / (a->invMass + b->invMass) * percent);
    if (a->invMass > 0) a->position += correction * a->invMass;
    if (b->invMass > 0) b->position -= correction * b->invMass;
}

void checkFloorCollisions(RigidBody3D& body, float floorY) {
    if (body.shape == SHAPE_BOX) {
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
    } else if (body.shape == SHAPE_SPHERE) {
        if (body.position.y - body.radius < floorY) {
            Contact c;
            c.bodyA = &body;
            static RigidBody3D floor({0, floorY, 0}, {100, 0.1f, 100}, 0.0f);
            c.bodyB = &floor;
            c.point = {body.position.x, floorY, body.position.z};
            c.normal = {0, 1.0f, 0};
            c.penetration = floorY - (body.position.y - body.radius);
            resolveContact(c);
        }
    }
}

// ==========================================
// 4. CAMERA, SHADOWS & PROJECTION ENGINE
// ==========================================

struct Camera {
    float angleX = 0.3f;
    float angleY = 0.0f;
    float distance = 11.0f;
    Vec3 target = {0.0f, -0.5f, 0.0f};

    Vec3 getPosition() const {
        return {
            target.x + distance * std::sin(angleY) * std::cos(angleX),
            target.y + distance * std::sin(angleX),
            target.z + distance * std::cos(angleY) * std::cos(angleX)
        };
    }
} camera;

struct SDL_Point2D { int x, y; };

SDL_Point2D project3DTo2D(Vec3 p, int screenW, int screenH) {
    float fov = 400.0f;
    Vec3 camPos = camera.getPosition();
    
    Vec3 forward = (camera.target - camPos).normalized();
    Vec3 right = forward.cross({0, 1, 0}).normalized();
    Vec3 up = right.cross(forward).normalized();

    Vec3 localP = p - camPos;
    Vec3 transP = {
        localP.dot(right),
        localP.dot(up),
        localP.dot(forward)
    };

    if (transP.z <= 0.1f) transP.z = 0.1f;

    int x = (int)((transP.x * fov) / transP.z + screenW / 2);
    int y = (int)((-transP.y * fov) / transP.z + screenH / 2);
    return {x, y};
}

// Projection Shadow Renderer (Flattens objects onto floor)
void drawShadow(SDL_Renderer* renderer, const RigidBody3D& body, float floorY, int screenW, int screenH) {
    SDL_SetRenderDrawColor(renderer, 35, 35, 45, 255); // Dark Shadow Color

    if (body.shape == SHAPE_BOX) {
        auto verts = body.getVertices();
        std::vector<SDL_Point2D> pts(8);
        for (int i = 0; i < 8; ++i) {
            Vec3 projVert = {verts[i].x, floorLevel + 0.02f, verts[i].z};
            pts[i] = project3DTo2D(projVert, screenW, screenH);
        }

        int edges[12][2] = {
            {0,1}, {1,2}, {2,3}, {3,0},
            {4,5}, {5,6}, {6,7}, {7,4},
            {0,4}, {1,5}, {2,6}, {3,7}
        };

        for (int i = 0; i < 12; ++i) {
            SDL_RenderDrawLine(renderer, 
                pts[edges[i][0]].x, pts[edges[i][0]].y,
                pts[edges[i][1]].x, pts[edges[i][1]].y
            );
        }
    } else { // Sphere Shadow Ring
        int segments = 10;
        for (int i = 0; i < segments; ++i) {
            float a1 = (i / (float)segments) * 2.0f * M_PI;
            float a2 = ((i + 1) / (float)segments) * 2.0f * M_PI;

            Vec3 s1 = {body.position.x + std::cos(a1) * body.radius, floorLevel + 0.02f, body.position.z + std::sin(a1) * body.radius};
            Vec3 s2 = {body.position.x + std::cos(a2) * body.radius, floorLevel + 0.02f, body.position.z + std::sin(a2) * body.radius};

            auto p1 = project3DTo2D(s1, screenW, screenH);
            auto p2 = project3DTo2D(s2, screenW, screenH);

            SDL_RenderDrawLine(renderer, p1.x, p1.y, p2.x, p2.y);
        }
    }
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

    SDL_SetRenderDrawColor(renderer, 0, 255, 120, 255);
    for (int i = 0; i < 12; ++i) {
        SDL_RenderDrawLine(renderer, 
            pts[edges[i][0]].x, pts[edges[i][0]].y,
            pts[edges[i][1]].x, pts[edges[i][1]].y
        );
    }
}

void drawWireframeSphere(SDL_Renderer* renderer, const RigidBody3D& body, int screenW, int screenH) {
    SDL_SetRenderDrawColor(renderer, 255, 180, 0, 255);
    int segments = 12;
    Mat3 R = body.orientation.toMatrix();

    for (int i = 0; i < segments; ++i) {
        float a1 = (i / (float)segments) * 2.0f * M_PI;
        float a2 = ((i + 1) / (float)segments) * 2.0f * M_PI;

        Vec3 c1_1 = body.position + R * Vec3{std::cos(a1) * body.radius, std::sin(a1) * body.radius, 0};
        Vec3 c1_2 = body.position + R * Vec3{std::cos(a2) * body.radius, std::sin(a2) * body.radius, 0};

        Vec3 c2_1 = body.position + R * Vec3{0, std::cos(a1) * body.radius, std::sin(a1) * body.radius};
        Vec3 c2_2 = body.position + R * Vec3{0, std::cos(a2) * body.radius, std::sin(a2) * body.radius};

        auto p1_1 = project3DTo2D(c1_1, screenW, screenH);
        auto p1_2 = project3DTo2D(c1_2, screenW, screenH);
        auto p2_1 = project3DTo2D(c2_1, screenW, screenH);
        auto p2_2 = project3DTo2D(c2_2, screenW, screenH);

        SDL_RenderDrawLine(renderer, p1_1.x, p1_1.y, p1_2.x, p1_2.y);
        SDL_RenderDrawLine(renderer, p2_1.x, p2_1.y, p2_2.x, p2_2.y);
    }
}

void drawFloorGrid(SDL_Renderer* renderer, float floorY, int screenW, int screenH) {
    SDL_SetRenderDrawColor(renderer, 60, 60, 80, 255);
    for (float x = -5.0f; x <= 5.0f; x += 1.0f) {
        auto p1 = project3DTo2D({x, floorY, -5.0f}, screenW, screenH);
        auto p2 = project3DTo2D({x, floorY, 5.0f}, screenW, screenH);
        SDL_RenderDrawLine(renderer, p1.x, p1.y, p2.x, p2.y);
    }
    for (float z = -5.0f; z <= 5.0f; z += 1.0f) {
        auto p1 = project3DTo2D({-5.0f, floorY, z}, screenW, screenH);
        auto p2 = project3DTo2D({5.0f, floorY, z}, screenW, screenH);
        SDL_RenderDrawLine(renderer, p1.x, p1.y, p2.x, p2.y);
    }
}

// ==========================================
// 5. MAIN LOOP WITH TAP-TO-SPAWN & PUSH
// ==========================================

float floorLevel = -2.0f;

int main(int argc, char* argv[]) {
    SDL_Init(SDL_INIT_VIDEO);

    int screenW = 720, screenH = 1280;
    SDL_Window* window = SDL_CreateWindow("3D Engine: Interactive Shadows & Stacking", 
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 
        screenW, screenH, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    std::vector<RigidBody3D> world;

    // Stacked Structure Initial Setup
    world.push_back(RigidBody3D({0.0f, floorLevel + 0.5f, 0.0f}, {0.5f, 0.5f, 0.5f}, 2.0f, SHAPE_BOX));
    world.push_back(RigidBody3D({0.0f, floorLevel + 1.6f, 0.0f}, {0.5f, 0.5f, 0.5f}, 1.5f, SHAPE_BOX));

    Vec3 gravity{0.0f, -9.81f, 0.0f};
    float dt = 0.016f;
    bool running = true;
    SDL_Event e;

    bool isDragging = false;
    int prevMouseX = 0, prevMouseY = 0;
    int touchStartX = 0, touchStartY = 0;

    while (running) {
        SDL_GetRendererOutputSize(renderer, &screenW, &screenH);

        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = false;

            if (e.type == SDL_MOUSEBUTTONDOWN) {
                isDragging = true;
                prevMouseX = e.button.x;
                prevMouseY = e.button.y;
                touchStartX = e.button.x;
                touchStartY = e.button.y;
            } else if (e.type == SDL_MOUSEBUTTONUP) {
                isDragging = false;
                
                // If quick tap (not drag), spawn an object or push existing ones
                int dragDist = std::abs(e.button.x - touchStartX) + std::abs(e.button.y - touchStartY);
                if (dragDist < 10) {
                    ShapeType type = (rand() % 2 == 0) ? SHAPE_BOX : SHAPE_SPHERE;
                    Vec3 camPos = camera.getPosition();
                    
                    // Launch new projectile from camera direction
                    Vec3 launchDir = (camera.target - camPos).normalized();
                    float spawnX = ((e.button.x / (float)screenW) - 0.5f) * 3.0f;
                    
                    RigidBody3D newBody({spawnX, 2.5f, 0.0f}, {0.4f, 0.4f, 0.4f}, 1.5f, type);
                    newBody.linearVelocity = launchDir * 6.0f + Vec3{0, 2.0f, 0};
                    newBody.angularVelocity = {2.0f, 3.0f, 1.0f};
                    world.push_back(newBody);
                }
            } else if (e.type == SDL_MOUSEMOTION && isDragging) {
                int deltaX = e.motion.x - prevMouseX;
                int deltaY = e.motion.y - prevMouseY;
                
                camera.angleY -= deltaX * 0.005f;
                camera.angleX += deltaY * 0.005f;
                camera.angleX = std::clamp(camera.angleX, -1.4f, 1.4f);

                prevMouseX = e.motion.x;
                prevMouseY = e.motion.y;
            }
        }

        // --- Physics Integration ---
        for (auto& body : world) {
            body.applyForceAtPoint(gravity * body.mass, body.position);
            body.integrate(dt);
            checkFloorCollisions(body, floorLevel);
        }

        // --- Render Step ---
        SDL_SetRenderDrawColor(renderer, 20, 20, 25, 255);
        SDL_RenderClear(renderer);

        drawFloorGrid(renderer, floorLevel, screenW, screenH);

        // Draw Shadows First (Below Objects)
        for (const auto& body : world) {
            drawShadow(renderer, body, floorLevel, screenW, screenH);
        }

        // Draw 3D Objects
        for (const auto& body : world) {
            if (body.shape == SHAPE_BOX) {
                drawWireframeBox(renderer, body, screenW, screenH);
            } else {
                drawWireframeSphere(renderer, body, screenW, screenH);
            }
        }

        SDL_RenderPresent(renderer);
        SDL_Delay(16);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
