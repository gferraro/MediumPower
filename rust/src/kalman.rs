/// Pure-Rust constant-velocity Kalman filter matching the OpenCV KalmanFilter
/// configuration used in the Python code:
///
///   dynamParams  = 4  (x, y, vx, vy)
///   measureParams = 2  (x, y)
///   transitionMatrix  = [[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]
///   measurementMatrix = eye(2,4)
///   processNoiseCov   = eye(4) * 0.03
///   measurementNoiseCov = eye(2)          (OpenCV default)
///   errorCovPost      = eye(4)            (OpenCV default)

type M4 = [[f32; 4]; 4];
type M2 = [[f32; 2]; 2];
type V4 = [f32; 4];
type V2 = [f32; 2];

fn mat4_mul(a: &M4, b: &M4) -> M4 {
    let mut r = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                r[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    r
}

fn mat4_add(a: &M4, b: &M4) -> M4 {
    let mut r = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            r[i][j] = a[i][j] + b[i][j];
        }
    }
    r
}

fn mat4_sub(a: &M4, b: &M4) -> M4 {
    let mut r = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            r[i][j] = a[i][j] - b[i][j];
        }
    }
    r
}

fn mat4_transpose(a: &M4) -> M4 {
    let mut r = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            r[j][i] = a[i][j];
        }
    }
    r
}

/// Invert a 2×2 matrix.
fn mat2_inv(a: &M2) -> M2 {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    let d = 1.0 / det;
    [
        [ a[1][1] * d, -a[0][1] * d],
        [-a[1][0] * d,  a[0][0] * d],
    ]
}

// H = [[1,0,0,0],[0,1,0,0]]   (first 2 rows of identity)
// H * x  → [x[0], x[1]]
// H * P  → first 2 rows of P
// P * Hᵀ → first 2 columns of P

pub struct Kalman {
    /// State vector [x, y, vx, vy].
    x: V4,
    /// Error covariance matrix (4×4).
    p: M4,
}

const F: M4 = [
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

const Q: M4 = [
    [0.03, 0.0,  0.0,  0.0 ],
    [0.0,  0.03, 0.0,  0.0 ],
    [0.0,  0.0,  0.03, 0.0 ],
    [0.0,  0.0,  0.0,  0.03],
];

/// Initial error covariance (identity).
const P0: M4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

impl Kalman {
    pub fn new() -> Self {
        Self { x: [0.0; 4], p: P0 }
    }

    /// Predict the next state.  Returns `(predicted_x, predicted_y)`.
    /// This matches `kalman.predict()` in Python, which returns a 4×1 matrix
    /// where `result[0][0]` is x and `result[1][0]` is y.
    pub fn predict(&mut self) -> (f32, f32) {
        // x_pred = F * x
        let fx = [
            F[0][0]*self.x[0] + F[0][2]*self.x[2],
            F[1][1]*self.x[1] + F[1][3]*self.x[3],
            self.x[2],
            self.x[3],
        ];
        // P_pred = F * P * Fᵀ + Q
        let fp  = mat4_mul(&F, &self.p);
        let ft  = mat4_transpose(&F);
        let fft = mat4_mul(&fp, &ft);
        let p_pred = mat4_add(&fft, &Q);

        self.x = fx;
        self.p = p_pred;

        (self.x[0], self.x[1])
    }

    /// Update state with a measurement (centroid x, centroid y).
    pub fn correct(&mut self, centroid: [f32; 2]) {
        // Innovation:  y = z - H*x  (H selects first 2 components)
        let innov: V2 = [centroid[0] - self.x[0], centroid[1] - self.x[1]];

        // S = H * P * Hᵀ + R   (top-left 2×2 of P plus identity R)
        let s: M2 = [
            [self.p[0][0] + 1.0, self.p[0][1]],
            [self.p[1][0],       self.p[1][1] + 1.0],
        ];
        let s_inv = mat2_inv(&s);

        // K = P * Hᵀ * S⁻¹   (Hᵀ selects first 2 columns of P)
        // P * Hᵀ → 4×2 matrix from first 2 columns of P
        let ph: [[f32; 2]; 4] = [
            [self.p[0][0], self.p[0][1]],
            [self.p[1][0], self.p[1][1]],
            [self.p[2][0], self.p[2][1]],
            [self.p[3][0], self.p[3][1]],
        ];
        // K = PH * S_inv  (4×2)
        let mut k = [[0.0f32; 2]; 4];
        for i in 0..4 {
            for j in 0..2 {
                for m in 0..2 {
                    k[i][j] += ph[i][m] * s_inv[m][j];
                }
            }
        }

        // x = x + K * innov
        for i in 0..4 {
            self.x[i] += k[i][0] * innov[0] + k[i][1] * innov[1];
        }

        // P = (I - K*H) * P
        // K*H is 4×4 where first 2 columns are K, rest are 0
        let mut kh = [[0.0f32; 4]; 4];
        for i in 0..4 {
            kh[i][0] = k[i][0];
            kh[i][1] = k[i][1];
        }
        let i_kh = mat4_sub(&P0, &kh);
        self.p = mat4_mul(&i_kh, &self.p);
    }
}
