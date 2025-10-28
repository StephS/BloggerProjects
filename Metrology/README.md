**Critical Coordinate System Distinction**

**[CRITICAL COORDINATE SYSTEM NOTE]**
This book uses two related but distinct angular reference systems that must never be confused:

**Indicator Positioning**: 
- Measured COUNTERCLOCKWISE from +Y axis (top position)
- 0° = +Y (top), 90° = +X (left), 180° = -Y (bottom), 270° = -X (right)
- Used for $\theta_\text{ind}$ in all calculations

**Surface Labeling**:
- Measured CLOCKWISE from top when viewing from operator position while the spindle is indexed at its 0° starting position ($\theta_\text{spin} = 0$)
- A = 0° (top), B = 90° (right), C = 180° (bottom), D = 270° (left)
- Used for surface identification and $\theta_\text{surf}$
- The surfaces rotate with the spindle, requiring re-calculating the surface positions.

**CRITICAL CONVERSION REQUIREMENT**:
When calculating which surface an indicator sees, you MUST convert from CCW indicator coordinates to CW surface coordinates using:

**Surface Notation**: $S_{\theta_\text{surf}}$ where:
- $\theta_\text{surf} = (360° - (\theta_\text{ind} - \theta_\text{spin})) \mod 360°$
  - Note: The subtraction from 360° converts from CCW indicator coordinates to CW surface coordinates
- $\theta_\text{ind}$ = indicator angular position (CCW from +Y axis)
- $\theta_\text{spin}$ = spindle rotation angle (CCW)

**Full Model Equation**:

$$I_{(\theta_\text{ind}, \theta_\text{spin})}(z_i) = M_x\cdot\sin(\theta_\text{ind})(z_i) + M_y\cdot\cos(\theta_\text{ind})(z_i) + S_{\theta_\text{surf}}(z_i) + T_{(\theta_\text{ind},\theta_\text{spin})}(z_i)$$

$$T_{(\theta_\text{ind},\theta_\text{spin})} = \alpha_x\cdot\sin(\theta_\text{ind}) + \alpha_y\cdot\cos(\theta_\text{ind}) + \beta_x\cdot\sin(\theta_\text{surf}) + \beta_y\cdot\cos(\theta_\text{surf}) + \gamma$$

**Surface Forms**:
The deviation of the surfaces from a straight line 

| Symbol | Definition | Units | Description |
|--------|------------|-------|-------------|
| $M_x(z_i)$ | Machine form error in X | unitless | Horizontal straightness profile |
| $M_y(z_i)$ | Machine form error in Y | unitless | Vertical straightness profile |
| $S_{\theta_\text{surf}}(z_i)$ | Artifact form error at surface $\theta_\text{surf}$ | unitless | Artifact straightness profile |

surfaces are denoted by their angular position $S_{\theta_\text{surf}}(z_i)$

**Machine Tilt ($\alpha$) (alpha) - Always 2 Components**

| Symbol | Definition | Units | Description |
|--------|------------|-------|-------------|
| $\alpha_x$ | Machine slope in X-Z plane | unitless | Slope between guideway and spindle axis (yaw) |
| $\alpha_y$ | Machine slope in Y-Z plane | unitless | Slope between guideway and spindle axis (pitch) |

**1. Machine Tilt $\alpha$ - Fixed to Machine Frame**
- **Physical Nature**: Misalignment between machine guideway and spindle axis
- **Mathematical Representation**: $\alpha_x$ (yaw), $\alpha_y$ (pitch)
- **Key Behavior**: Does NOT rotate with spindle
- **In Equations**: Coefficient changes ONLY with indicator position
- **Measurement Implication**: Affects all indicators based on their position relative to machine frame
- **Physical Reasoning**: Machine frame is stationary; the guideway-to-spindle misalignment remains fixed regardless of spindle rotation

**2. Setup Tilt $\beta$ - Rotates with Spindle**
- **Physical Nature**: Misalignment between spindle axis and artifact centerline
- **Mathematical Representation**: $\beta_x$, $\beta_y$ in reference to the spindle position at zero degrees.
- **Key Behavior**: Rotates WITH the spindle/artifact assembly
- **In Equations**: Coefficient changes with BOTH indicator position AND spindle rotation
- **Measurement Implication**: ONLY tilt affected by spindle rotation
- **Physical Reasoning**: When artifact rotates 180°, the high side becomes the low side, reversing tilt direction

**3. Artifact Tilt $\gamma$ - Intrinsic Cone Geometry**
- **Physical Nature**: Cone angle of the artifact itself
- **Mathematical Representation**: $\gamma$
- **Key Behavior**: Intrinsic property that doesn't change with rotation
- **In Equations**: Coefficient doesn't change
- **Measurement Implication**: The cone shape is the same for all surfaces. It's an imaginary tilt that is the average of all measurements.
- **Physical Reasoning**: The cone geometry is fixed to the artifact; rotation doesn't change the cone angle
