# **A Unified Method for Form and Tilt Measurement**
This document outlines a powerful, unified metrology method for simultaneously determining the **form** (profile error) and **tilt** (parallelism error) of a machine slideway and a test artefact. It combines the principles of the "Classic Straightedge Reversal" and the "Spindle Parallelism Check" into a single, comprehensive workflow.
This technique is a variant of square reversal designed to determine the parallelism error between a spindle's axis of rotation and a machine's linear slideway. Its main advantage is that it can isolate the true machine geometry error even when using an imperfect test artefact and accounting for setup errors.
The process involves mounting the imperfect cylindrical square on the spindle and using a displacement indicator to trace its surface as the slideway moves. The core of the method is a sequence of four measurements where the signs of the different error components ( $\alpha, \beta, \gamma$ ) are systematically manipulated through physical reversals.

## **Core Purpose**
The primary goal of this method is to solve one of the fundamental problems in precision metrology: separating the geometric errors of the measuring instrument from the errors of the object being measured. It does this by simultaneously solving for six key error components: three form profiles and three tilt angles. By decomposing each measurement into its linear (tilt) and non-linear (form) components, this method allows for a complete characterization of the system's geometric errors from 4 sets of measurements.

---
## **1. Core Concepts: Form vs. Tilt**
Imagine measuring a long, slightly wavy road built on the side of a hill.
* **Tilt (Parallelism Error)**: This is the overall angle of the hill itself. It's a single, constant slope that affects the entire road. In our model, this is represented by the constant angular errors $\alpha$, $\beta$, and $\gamma$.
* **Form (Profile Error)**: This is the waviness or bumpiness of the road surface, independent of the hill it's on. In our model, this is the error profile represented by the functions $M(x)$ and $S(x)$.

The unified method is designed to measure both the "hill" and the "waviness" at the same time.

---
## **2. Setup and Methodology**
The physical setup and measurement procedure are crucial for acquiring the necessary data.

### **Setup**
* A cylindrical artefact (the "square") is mounted onto the machine's spindle.
* A high-resolution displacement indicator (referred to as a Sensor) is mounted on a carriage that travels along the machine's slideway (the X-axis).
* A method to indicate the spindle accurately at 0 and 180 degrees. Ideally every 90 degrees to get both axis.
* The setup must allow for the indicator to trace the surface of the artefact and for the spindle to be rotated precisely 180°.
* The indicators must be aligned properly for best accuracy. There are two indicator positions: Top and Bottom (or if measuring horizontally, Front and Back). There is a process that can be followed to align the sensor with the axis.

### **The Measurement Setup: Two Indicators, Two Surfaces**
The physical setup requires two separate displacement indicators fixed to the machine's moving carriage. Their positions are fixed relative to each other and do not change during the procedure. Alternatively a single indicator can be used, but it requires precise positioning to be performed twice. Knowing which indicator traces which surface is fundamental to how the signs in the equations are derived, and without this clarity, the method is impossible to implement correctly. The two indicators measure the machine error, $M(x)$, with opposite signs because they are on opposite sides of the centerline.

* **Near Indicator**: The indicator positioned closer to the machine slideway. Example: while measuring the vertical axis on a lathe, the bottom indicator is the Near indicator.
* **Far Indicator**: The indicator positioned further from the machine slideway. Example: while measuring the vertical axis on a lathe, the top indicator is the Far indicator.
* **Surface A & Surface B**: The two distinct surfaces of the test artefact being measured. Side A and side B are determined by your initial measurement. With the spindle at the 0° home position, the Far indicator traces Surface A, while the Near indicator traces Surface B.
**Another way to put it** after calculating the resulting tilt, the Far indicator is the positive tilt direction, the Near indicator is the negative tilt direction.

### **Methodology: The Four-Measurement Reversal**
The entire process is built upon a series of four measurements taken in two spindle positions. The key is that the 180° rotation of the spindle swaps which surface is presented to each fixed indicator.

1.  **Spindle Position 1 (0°)** spindle at its 0° home position
    * **Measurement $I_{1A}(x)$**: The **Near Indicator** traces **Surface A**.
    * **Measurement $I_{1B}(x)$**: The **Far Indicator** traces **Surface B**. This reverses the effect of the artefact's own imperfection ( $\gamma$ ).

2.  **Spindle Position 2 (180°)** spindle is rotated precisely 180°.
    * **Measurement $I_{2B}(x)$**: The **Near Indicator** now traces the flipped **Surface B**. This reverses all three tilt components.
    * **Measurement $I_{2A}(x)$**: The **Far Indicator** now traces the flipped **Surface A**. This reverses the effect of the machine error ( $\alpha$ ) and the setup tilt error ( $\beta$ ).

**Note:** After the spindle rotation the surfaces measured by the indicators are swapped.

---
## **3. The Mathematical Model**
The entire method is built on a model that separates the form and tilt components for each measurement.

### **Definitions of Terms, Symbols, and Variables**
The following measurements use the subscript ${ij}$ where:

**Configuration**: $i = \{1,2\}$
- 1 is the forward (0°) position
- 2 is the reversed (180° flipped) position

**Artefact Side**: $j = \{A,B\}$
- A is the top side of the artefact before reversal
- B is the bottom side of the artefact before reversal

**Error components:** (unknowns)
* **$M(x)$**: The machine slide's **form** error profile  as a function of position $x$
* **$S_j(x)$**: The artefact's **form** error profile.
    * **$S_A(x)$**: The artefact's **form** error profile of side A.
    * **$S_B(x)$**: The artefact's **form** error profile of side B (180 degrees opposite to side A).
* **$\alpha$ (alpha)**: The primary value being measured. It represents the parallelism error, or the angle of misalignment, between the machine's spindle axis and its slideway axis. The machine's constant parallelism **tilt** error.
* **$\beta$ (beta)**: A setup error representing the angle that the test artefact's central axis is tilted with respect to the spindle's actual axis of rotation. This error occurs during mounting. The constant setup **tilt** error.
* **$\gamma$ (gamma)**: The inherent imperfection of the test artefact. The paper models the "square" as the frustum of an oblique cone, where $\gamma$ is the tilt angle of the cone's generator. This means the surface being measured is not a perfect cylinder. The artefact's inherent constant **tilt** error.

**Measurement Channels**:
* **$I_{ij}$**: The **measurement** channels obtained from the measurement process. These are single-point x-y pairs.
    <details>
    <summary>All the measurement channels</summary>

    * $I_{1A}(x)$
    * $I_{1B}(x)$
    * $I_{2A}(x)$
    * $I_{2B}(x)$

    </details>

**Intermediates:**
* $T_{ij}$: These are the the **slopes** calculated from a least squares fit (linear regression) of the displacement indicator's output as the slideway moves along its path. The least squares fit of each measurement channel.
    <details>
    <summary>All the slopes</summary>

    * $T_{1A}(x)$
    * $T_{1B}(x)$
    * $T_{2A}(x)$
    * $T_{2B}(x)$

    </details>
<!-- -->
* $b_{ij}(x)$: Constant DC offset. Calculated by the best-fit line DC Offset (b term)
    <details>
    <summary>All the DC Offsets</summary>

    * $b_{1A}(x)$
    * $b_{1B}(x)$
    * $b_{2A}(x)$
    * $b_{2B}(x)$

    </details>
<!-- -->
* $C_{ij}(x)$: Zeroed Form profiles by removing the constant DC offset. Calculated by original data ( $I_{ij}$ ) minus the DC Offset ( $b_{ij}(x)$ term)
    <details>
    <summary>All the Zeroed Form profiles</summary>

    * $C_{1A}(x)$
    * $C_{1B}(x)$
    * $C_{2A}(x)$
    * $C_{2B}(x)$

    </details>
<!-- -->
* $R_{ij}(x)$: Form **profile residuals** by removing tilt through the chosen method (shearing or rotation). Calculated from the original data ( $I_{ij}$ )
    <details>
    <summary>All the profile residuals</summary>

    * $R_{1A}(x)$
    * $R_{1B}(x)$
    * $R_{2A}(x)$
    * $R_{2B}(x)$

    </details>
<!-- -->
* **$F_1(x)$, $F_2(x)$**: The **averaged** Form **profile residuals**. Taken by averaging ( $R_{1j}(x)$ ) and ( $R_{2j}(x)$ ) **Note** Might be optional?

### **The Four Measurement Equations**
Each raw measurement is the sum of the combined form profile and the combined tilt profile.
First, we redefine our measurement outputs ( $I$ ) as functions of position $x$. Each function contains both a form component (the profile) and a linear component (the tilt or parallelism error multiplied by $x$ ).
You can create a system of linear equations that can be easily solved to isolate each error component.
Using the same four-measurement setup, the resulting functions would be:
  
| **Spindle Position 1 (0°)** spindle at its 0° home position | | |
| --- | --- | --- |
| $I_{1A}(x) = [M(x) + S_A(x)] + (\alpha + \beta + \gamma) \cdot x$ | | (Near Indicator, 0°) |
| $I_{1B}(x) = [-M(x) + S_B(x)] + (-\alpha + \beta - \gamma) \cdot x$ | | (Far Indicator, 0°) |

| **Spindle Position 2 (180°)** spindle is rotated precisely 180° | | |
| --- | --- | --- |
| $I_{2B}(x) = [M(x) + S_B(x)] + (\alpha - \beta - \gamma) \cdot x$ | | (Near Indicator, 180°) |
| $I_{2A}(x) = [-M(x) + S_A(x)] + (-\alpha - \beta + \gamma) \cdot x$ | | (Far Indicator, 180°) |

***Note:*** In the reversed positions ( $I_{1B}$, $I_{2A}$ ), the machine profile $M(x)$ and its corresponding tilt $\alpha$ are inverted, as is the setup tilt $\beta$.

---
## **4. The Complete Workflow: From Raw Data to Final Results**
The process involves three main stages: separating the data, solving for tilt, and solving for form.

### **Stage 1: Separate Tilt From Form**
This stage uses linear regression to decompose each of the four raw data channels.
For each of the four raw measurement functions ( $I_{1A}(x)$ through $I_{2B}(x)$ ), you perform a linear regression (least-squares fit).

* The **slope** of the best-fit line gives you the combined tilt component. This yields four slope values, which we can call $T_{1A}, T_{1B}, T_{2A},$ and $T_{2B}$. For example, $T_{1A} = \alpha + \beta + \gamma$.
* The **DC Offset** from the best-fit line gives you the combined offset terms. The yields four offset values, which we can call $b_{1A}, b_{1B}, b_{2A},$ and $b_{2B}$

#### 1. **Perform Linear Regression**:

For each of the four data profiles ( $I_{ij}(x)$ ), calculate the best-fit line (least squares fit). The **slope** (aka m term) of the four best-fit line from the least squares calculations results in the four combined tilt components ( $T_{ij}$ ). We also need the constant (b term) to remove any offset.

#### 2. **Extract Tilt Components**: The **slope** of each best-fit line gives you the combined tilt values.
* $T_{1A} = \alpha + \beta + \gamma$
* $T_{1B} = -\alpha + \beta - \gamma$
* $T_{2A} = -\alpha - \beta + \gamma$
* $T_{2B} = \alpha - \beta - \gamma$

#### 3. **Extract DC Offset from Form: The Constant Term**

Given that the measurement's vertical offset is arbitrary and the true centerline is unknown, errors can occur if you don't remove the offset (depending on the rotation method you choose).
The best and most robust methodology is to **remove the the constant offset before performing the rotation.**

**Why This Is the Correct Approach**

* **The Arbitrary Offset is a Nuisance Variable**: As you said, the measurement's DC offset `b` is arbitrary. It depends on how the operator zeroed the indicator, not on the physics of the part. We need to remove its influence.
* **Rotation and Shape Distortion**: The physical rotation occurs around the spindle's axis. The *shape distortion* we are trying to model comes from the fact that different parts of the form profile (the "waviness" $S(x)$ ) have different heights and are therefore shifted laterally by different amounts.
* **Uniform vs. Differential Shift**: A large DC offset `b` (or the unknown distance to the centerline) only contributes a **uniform lateral shift** to the entire profile. It moves the whole curve left or right, but it does not change its *shape*. The shape distortion comes purely from the **differential shifts** caused by the varying height of the zero-centered form profile $S(x)$.

**The Most Robust Workflow**

1. **Isolate the Zero-Centered Form**: For each raw measurement (e.g., $I_{1A}(x)$ ), perform a linear regression to find the best-fit line $y_{fit} = \theta \cdot x + b$. The result is the zero-centered form profile: $C_{1A}(x) = I_{1A}(x) - b_{1A}$.
2. **Rotate the Zero-Centered Form**: Apply the rotation transformation **only to this form profile $F(x)$**. This correctly calculates the *shape distortion* caused by the rotation, completely independent of the unknown and arbitrary measurement offset.
3. **Proceed**: Use this correctly de-tilted form profile in all subsequent calculations.

This approach is superior because it is immune to arbitrary vertical offsets in the measurement setup, making it far more robust for real-world applications.

**Extract the zeroed offset terms**:
Remove the offset from the measured channels:
$$C_{ij}(x) = I_{ij}(x) - b_{ij}$$

<details>
<summary>All the calculations</summary>

* $C_{1A}(x) = I_{1A}(x) - b_{1A}$
* $C_{1B}(x) = I_{1B}(x) - b_{1B}$
* $C_{2A}(x) = I_{2A}(x) - b_{2A}$
* $C_{2B}(x) = I_{2B}(x) - b_{2B}$

</details>
<br>

At the end of Stage 1, you have successfully separated the problem into two parts: a set of four tilt values ( $T_{ij}$ ) and a set of 4 Offset Zeroed form profiles ( $C_{1A}(x), C_{1B}(x), C_{2A}(x), C_{2B}(x)$ ).

---
### **Stage 2: Solve for Tilt Components**
**Solving for Tilt:** 
You have four slope values ( $T_{1A}$ through $T_{2B}$ ). Using the four slope values from Stage 1, you can solve for each individual tilt angle:

$$T_{1A} = \alpha + \beta + \gamma$$
$$T_{1B} = -\alpha + \beta - \gamma$$
$$T_{2A} = -\alpha - \beta + \gamma$$
$$T_{2B} = \alpha - \beta - \gamma$$

**Update** I'm pretty sure you can use all 4 equations to solve for a single variable. This averages the error in each.

**Step 1: Calculate the Artefact Imperfection (Artefact Tilt) ( $\gamma$ )**
By subtracting equations taken at the same spindle position, you cancel $\alpha$ and $\beta$.

- **Calculation:**

    <details>
    <summary>All the calculation steps</summary>

    $$T_{1A} = \alpha + \beta + \gamma$$
    $$- T_{1B} = \alpha - \beta + \gamma$$
    $$T_{2A} = -\alpha - \beta + \gamma$$
    $$- T_{2B} = -\alpha + \beta + \gamma$$
    ---
    </details>


    $$T_{1A} - T_{1B} + T_{2A} - T_{2B} = \alpha + \beta + \gamma + \alpha - \beta + \gamma - \alpha - \beta + \gamma - \alpha + \beta + \gamma$$
- **Formula:**
    $$\gamma = \frac{T_{1A} - T_{1B} + T_{2A} - T_{2B}}{4}$$

**Step 2: Calculate the Setup Tilt Error ( $\beta$ )**
By subtracting equations taken with the same indicator ('A' or 'B') but at different spindle positions, you cancel $\alpha$ and $\gamma$.

- **Calculation:**

    <details>
    <summary>All the calculation steps</summary>

    $$T_{1A} = \alpha + \beta + \gamma$$
    $$T_{1B} = -\alpha + \beta - \gamma$$
    $$- T_{2A} = \alpha + \beta - \gamma$$
    $$- T_{2B} = -\alpha + \beta + \gamma$$
    ---
    </details>

    $$T_{1A} + T_{1B} - T_{2A} - T_{2B} = \alpha + \beta + \gamma - \alpha + \beta - \gamma + \alpha + \beta - \gamma - \alpha + \beta + \gamma$$
* **Formula:**
    $$\beta = \frac{T_{1A} + T_{1B} - T_{2A} - T_{2B}}{4}$$

**Step 3: Calculate the Machine Parallelism Error (Machine Tilt) ( $\alpha$ )**
By adding specific pairs of equations, you can cancel out both the setup error ( $\beta$ ) and the artefact error ( $\gamma$ ).

- **Calculation:**

    <details>
    <summary>All the calculation steps</summary>

    $$T_{1A} = \alpha + \beta + \gamma$$
    $$- T_{1B} = \alpha - \beta + \gamma$$
    $$- T_{2A} = \alpha + \beta - \gamma$$
    $$T_{2B} = \alpha - \beta - \gamma$$
    ---
    </details>

    $$T_{1A} - T_{1B} - T_{2A} + T_{2B} = \alpha + \beta + \gamma + \alpha - \beta + \gamma + \alpha + \beta - \gamma + \alpha - \beta - \gamma$$
- **Formula:**
    $$\alpha = \frac{T_{1A} - T_{1B} - T_{2A} + T_{2B}}{4}$$

---
### **Stage 3: Solve for Form Components**

#### 1. **Extract Form Residuals by Shearing or Rotation**:
The **residuals** of each fit (original data transformed by rotation or shearing) give you the form profiles.

1. Decide on the transformation method based on the acceptable error. See **Modeling Errors: A Detailed Guide to Shearing vs. Rotation**
2. For the most basic method (shearing), in a perfect, noise-free world, the math works out as follows:
    * $R_{1A}(x) = C_{1A}(x) - T_{1A} \cdot x = M(x) + S_A(x)$
    * $R_{1B}(x) = C_{1B}(x) - T_{1B} \cdot x = -M(x) + S_B(x)$
    * $R_{2A}(x) = C_{2A}(x) - T_{2A} \cdot x = -M(x) + S_A(x)$
    * $R_{2B}(x) = C_{2B}(x) - T_{2B} \cdot x = M(x) + S_B(x)$

**Note:** A Higher quality rotation would incorporate the nominal thickness of the artefact. Take a few thickness measurements of the artifact at 2 or more points and average them. See the section on 

#### 2. **Handle Redundancy by Averaging**:
In a real measurement with noise, $F_{1A}(x)$ and $F_{1B}(x)$ will be slightly different. By averaging these redundant profiles, we reduce the effect of random noise and get a better estimate of the true form. Because both $R_{1A}$ and $R_{1B}$ should nominally equal $M(x)+S(x)$, their average cancels half the noise.  Likewise for $R_{2A}$ and $R_{2B}$.

* **Final Form Profile 1:**
    $$F_1(x) = \frac{R_{1A}(x) + R_{1B}(x)}{2}$$
* **Final Form Profile 2:**
    $$F_2(x) = \frac{R_{2A}(x) + R_{2B}(x)}{2}$$

Using the two averaged form profiles, you can solve for the individual error profiles.

* **You have:**
    * $F_1(x) = M(x) + S(x)$
    * $F_2(x) = -M(x) + S(x)$
* **To find $M(x)$ (Machine Form):** Subtract $F_2(x)$ from $F_1(x)$.
    $$F_1(x) - F_2(x) = (M(x) + S(x)) - (-M(x) + S(x)) = 2M(x)$$
    $$M(x) = \frac{F_1(x) - F_2(x)}{2}$$
* **To find $S(x)$ (Artefact Form):** Add $F_1(x)$ and $F_2(x)$.
    $$F_1(x) + F_2(x) = (M(x) + S(x)) + (-M(x) + S(x)) = 2S(x)$$
    $$S(x) = \frac{F_1(x) + F_2(x)}{2}$$

**Note**: TODO: I think there is an issue here, as it averages the artefact side A and side B together. That's okay for tilt, but not okay if we want to extract both Side A and Side B of the artefact. _However_ it's rare that we care about the artefacts form (unless it is what we want to measure). Instead, we typically want to measure the guideway. If we were doing a straightedge reversal, we'd need to adjust the equations to isolate the Side A and Side B. In that case we'd likely adjust the averaging equations to solve for the sides, and not the positions (but don't quote me on this, I'm uncertain). Or maybe they aren't averaged at all? We'd calculate the forms directly from the residuals. We'd still have 4 equations:

$$R_{1A}(x) = M(x) + S_A(x)$$
$$R_{1B}(x) = -M(x) + S_B(x)$$
$$R_{2A}(x) = -M(x) + S_A(x)$$
$$R_{2B}(x) = M(x) + S_B(x)$$

and we'd solve for $S_A(x)$ and $S_B(x)$ first. We have two equations to solve:

**Step 1: Calculate artefact's form profile of side A ( $S_A(x)$ )**
- **Calculation:**

    <details>
    <summary>All the calculation steps</summary>

    $$R_{1A}(x) = M(x) + S_A(x)$$
    $$R_{2A}(x) = -M(x) + S_A(x)$$
    ---
    </details>

    $$R_{1A}(x) + R_{2A}(x) = M(x) + S_A(x) -M(x) + S_A(x)$$

- **Formula:**
    $$S_A(x) = \frac{R_{1A}(x) + R_{2A}(x)}{2}$$

**Step 2: Calculate artefact's form profile of side B ( $S_B(x)$ )**
- **Calculation:**

    <details>
    <summary>All the calculation steps</summary>

    $$R_{1B}(x) = -M(x) + S_B(x)$$
    $$R_{2B}(x) = M(x) + S_B(x)$$
    ---
    </details>

    $$R_{1B}(x) + R_{2B}(x) = M(x) + S_B(x) -M(x) + S_B(x)$$

- **Formula:**
    $$S_B(x) = \frac{R_{1B}(x) + R_{2B}(x)}{2}$$

**Step 3: Calculate machine form profile ( $M(x)$ )**

We can solve for M by combining all 4 equations (first inverting $R_{1B}(x)$ and $R_{2A}(x)$ equations)

- **Calculation:**

    <details>
    <summary>All the calculation steps</summary>

    $$R_{1A}(x) = M(x) + S_A(x)$$
    $$- R_{1B}(x) = M(x) - S_B(x)$$
    $$- R_{2A}(x) = M(x) - S_A(x)$$
    $$R_{2B}(x) = M(x) + S_B(x)$$
    ---
    </details>

    $$R_{1A}(x) - R_{1B}(x) - R_{2A}(x) + R_{2B}(x) = M(x) + M(x) + M(x) + M(x)$$

- **Formula:**
    $$M(x) = \frac{R_{1A}(x) - R_{1B}(x) - R_{2A}(x) + R_{2B}(x)}{4}$$

At least I think that's right (need to verify). This does two things: solves for both artefact surfaces and the guideway, while simultaneously averaging out the error of the guideway. Looking at it, it doesn't actually require any additional calculations, if I'm correct.

TODO: This can also be expanded upon if we decide to measure the horizontal and vertical axis at the same time. We'd have 8 measurement channels which would result in a lot of averaging depending on how many indicator positions we use. I might update this document to describe both methods later.

---

By following this unified procedure, you successfully use a single set of four measurements to extract all the desired information: the machine's parallelism error ( $\alpha$ ), the machine's form error ( $M(x)$ ), and the artefact's form errors ( $S_A(x)$ and $S_B(x)$ ), while also solving for the setup and artefact tilt errors ( $\beta$ and $\gamma$ ).

---
## **5. Practical Considerations: Quantifying and Minimizing Error**
The accuracy of the results depends on understanding and managing the sources of error.

### **Systematic Errors: Imperfect Reversal**
Real-world mechanical reversals are never perfect. Unintended shifts ( $\Delta x$ ) during the 180° rotation can introduce errors that are a primary limitation of the method's accuracy. These are minimized with high-quality fixtures but must be considered in the final uncertainty budget.

TODO: show the equation for the shift

---
### **Modeling Errors: A Detailed Guide to Shearing vs. Rotation**
After acquiring the raw data, the first crucial step is to separate the linear tilt from the non-linear form. The choice of *how* you remove this tilt is a critical modeling decision. The choice between them depends on a trade-off between approximation error and interpolation error. This guide provides a detailed breakdown of the two methods, how to calculate their respective errors, and a clear framework for deciding which to use. There are two distinct formulas for the two error sources, assuming linear interpolation for the rotation method.

The core choice is between:
* **Shearing (Approximation)**: Computationally simple but physically an approximation. The error from this method comes from the **small angle approximation**.
* **Rotation (Physically Correct)**: Physically exact but requires a complex interpolation step that introduces its own error. This method applies a physically correct rigid body rotation to the data and then uses interpolation to place it back on the original measurement grid. Different interpolation methods can be chosen on implementation, which might provide some additional resistance to noise.

#### **Method 1: The Shearing Method**

This method "shears" the profile by subtracting the best-fit linear trend from the data.

##### **Methodology**
1.  A best-fit line, $y_{fit} = \theta \cdot x + b$, is calculated for a raw data profile (e.g., $I_{1A}(x)$ ). The slope, $\theta$, represents the combined tilt.
2.  This line is subtracted from the original data to get the form profile: $F(x) = I_{1A}(x) - y_{fit} - b$.

#### **Method 2: The Rotation Method**

This method applies a physically correct rigid body rotation to the data and then uses interpolation to place it back on the original measurement grid.

##### **Methodology**
1.  The overall tilt angle, $\theta$, is calculated.
2.  A rotation matrix is applied to every data point $(x, y)$ to get a new set of coordinates $(x', y')$.
3.  Since the new points $(x', y')$ are no longer on the original, evenly spaced x-grid, an **interpolation** scheme (e.g., linear or spline) is used to estimate the y-values at the original x-positions.
TODO: Add notes on thickness calculations used during rotation transformation

---
#### **Calculating Modeling Errors***

#### **Step 1: Perform Initial Data Separation**
* Run the **Stage 1** analysis on your raw data (e.g., $I_{1A}(x)$ ).
* This gives you the necessary inputs for the error analysis:
    * **$\theta$ (Overall Tilt Angle):**
        * **Description:** The main slope of the measured profile, in radians.
        * **How to get it:** This is the slope of the best-fit line calculated during the linear regression of your raw data (e.g., $T_{1A}$ ).
    * The initial form profile, **$F(x)$** (the residuals of the fit).

#### **Shearing Error: A Systematic Approximation Error**
This error is the result of using a linear "shear" to approximate a circular "rotation." The primary error is the vertical slump from the rotational arc, which is a predictable, systematic error. If the shear error is significantly smaller than other known sources of noise in your system, the shearing method is acceptable and preferred for its simplicity. This is the most common outcome in high-precision metrology.

* **Shearing Error Estimation Formula:**
    $$Error_{shear}(x) \approx -S(x) \cdot \frac{\theta^2}{2}$$
* **What this means:**
    * The error at any point $x$ is proportional to the height of the form profile $S(x)$ at that point. Tallest peaks and deepest valleys have the most error.
    * The error is proportional to the **square of the tilt angle ( $\theta^2$ )**. This is crucial—it tells us that this error shrinks extremely fast as the tilt gets smaller, making it negligible in most high-precision cases.

#### **Rotation Error: A Data-Dependent Interpolation Error**
This error occurs when you resample the correctly rotated data back onto the original grid. For linear interpolation, the error is greatest where the profile is most curved (i.e., has the largest second derivative). If you require the highest possible physical fidelity for a measurement with large angles, then the more complex rotation method is justified.

* **Rotation Error Estimation Formula (for Linear Interpolation):**
    $$Error_{interp}(x) \approx \frac{(\Delta x)^2}{8} \left| \frac{d^2S}{dx^2} \right|_{x}$$
    where the lateral shift is $\Delta x \approx S(x) \cdot \theta$. Substituting this in gives:
    $$Error_{interp}(x) \approx \frac{(S(x) \cdot \theta)^2}{8} \left| \frac{d^2S}{dx^2} \right|_{x}$$
* **What this means:**
    * This error is also proportional to the **square of the tilt angle ( $\theta^2$ )** and the **square of the feature height ( $S(x)^2$ )**.
    * Critically, it depends on the **local curvature ( $\frac{d^2S}{dx^2}$ )**. A perfectly straight or gently sloping profile has very little interpolation error, even with a large shift. A profile with sharp, tight curves will have a large interpolation error.

---
#### **2. Beyond Worst-Case: A Statistical Approach to Error**

Worst-case analysis can be misleading if driven by a single outlier. A statistical approach gives a much better sense of the overall impact of these errors.

Instead of just comparing the maximum error values, the more robust method is to compare their **Root Mean Square (RMS)** values. The RMS gives a measure of the average magnitude of the error across the entire profile.

#### **The Methodology**
1.  **Calculate the Full Error Profiles**: After you've solved for the form profiles ( $M(x)$, $S_A(x)$, $S_B(x)$ ) and the tilt angle $\theta$, use the more accurate formulas above to calculate the estimated error *for every point x* in your dataset. This gives you two new data vectors: $Error_{shear}(x)$ and $Error_{interp}(x)$.
2.  **Calculate the RMS of Each Error Profile**: The RMS is calculated as:
    $$RMS = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (error_i)^2}$$
    Calculate $RMS_{shear}$ and $RMS_{interp}$.
3.  **Compare**: The method with the lower RMS error is likely the better choice for your dataset, as it introduces less average error across the entire measurement. This approach is much more stable and less sensitive to single-point outliers than a worst-case comparison. 

---
### **The Physical Effect: Lateral Shift Due to Artefact Thickness**

When an artefact with a finite thickness (or diameter), **D**, is tilted by an angle, **$\theta$**, the top and bottom surfaces shift laterally relative to each other.

  * The top surface (Side A) is shifted horizontally from the centerline.
  * The bottom surface (Side B) is shifted in the opposite direction.

The magnitude of this relative lateral shift, **$\Delta x$**, between the measurement points on surface A and surface B is approximately:

$$\Delta x \approx D \cdot \sin(\theta)$$

For the very small angles in metrology, this simplifies to:

$$\Delta x \approx D \cdot \theta$$

#### **Why This Is a Problem**

This means that when the slideway is at position $x$, the indicator measuring side A probes a different physical point on the artefact's coordinate system than the indicator measuring side B. This violates the simple model's assumption that $S_A(x)$ and $S_B(x)$ are being evaluated at the same location.

The model for the second measurement, for instance, should more accurately be written as:

$$I_{1B}(x) = M(x) + S_B(x - \Delta x) + (\text{tilt terms}) \cdot x$$

The error introduced by ignoring this effect is approximately the product of the local slope of surface B and this lateral shift:

$$Error_{thickness} \approx \frac{dS_B}{dx} \cdot \Delta x \approx \frac{dS_B}{dx} \cdot (D \cdot \theta)$$

This error becomes significant when you have a combination of:

  * A **thick artefact** (large $D$ ).
  * A **large tilt angle** ( $\theta$ ).
  * A **complex surface form** (large local slope $\frac{dS_B}{dx}$ ).

#### **How to use it in the calculations**

It can only be used in methods where real world constraints are applied. That means it can't be used in shearing. But it can be used in rotation.
After removing the DC Offset from the measurements, you *Add* half the thickness of the artifact to the far side measurements, and *Subtract* half the thickness of the artifact to the near side measurements. Then you perform the rotation around a point. Afterwards you remove the same amount that you added. This should result in a fully constrained, physically accurate rotation.

---
## **6. The Ultimate Method**
### **The Simultaneous Solution: Nonlinear Least-Squares**

Instead of separating the problem into stages (tilt removal, then form separation), this method treats it as a single, large optimization problem.

#### **The Core Idea**
The goal is to find one comprehensive set of all parameters—the tilt angles and the coefficients describing the form profiles—that, when plugged into the measurement model, collectively minimizes the error (the sum of squared differences) across all four raw data channels simultaneously.

#### **A More Complete Model**
This approach allows for a more physically accurate model. As you noted, the two sides of the artefact may have different form errors. We can define them as two separate profiles, **$S_A(x)$** and **$S_B(x)$**.

The model for the four measurements becomes:
1. **Near Indicator, 0°:** measures the machine and side A:
    $$I_{1A}(x) = [M(x) + S_A(x)] + (\alpha + \beta + \gamma) \cdot x$$
2. **Far Indicator, 0°:** measures the reversed machine and side B:
    $$I_{1B}(x) = [-M(x) + S_B(x)] + (-\alpha + \beta - \gamma) \cdot x$$
3. **Far Indicator, 180°:** measures the reversed machine and side A:
    $$I_{2A}(x) = [-M(x) + S_A(x)] + (-\alpha - \beta + \gamma) \cdot x$$
4. **Near Indicator, 180°:** measures the machine and side B:
    $$I_{2B}(x) = [M(x) + S_B(x)] + (\alpha - \beta - \gamma) \cdot x$$

To make this work in a solver, the continuous functions ( $M(x)$, $S_A(x)$, $S_B(x)$ ) must be **parameterized** — represented by a set of coefficients for a basis function, like a polynomial or a series of splines. The solver then finds the best values for these coefficients alongside the best values for the tilt angles $\alpha, \beta, \text{ and } \gamma$.

---
### **The Accuracy Advantage: A Holistic Approach**

A simultaneous nonlinear least-squares fit can be more accurate than the staged shear or rotation methods for one primary reason: **it avoids compounding errors.**

* **The Staged Method's Weakness**: The two-stage approach makes a permanent decision in Stage 1 when it removes the tilt. Any error introduced in that first step—whether from the shearing approximation or from interpolation after rotation—is irrevocably "baked into" the residual data. The second stage (form separation) has no way to correct for it.

* **The Simultaneous Method's Strength**: A nonlinear solver considers all parameters at once in a **holistic optimization**. It is free to trade errors between parameters to find the best global solution. For instance, it might determine that a slightly different value for the tilt angle $\alpha$ allows the form profiles $M(x)$ and $S_A(x)$ to fit the data much more closely. This coupling allows it to find a better overall compromise that minimizes the total error, something a staged approach cannot do.

Furthermore, the solver's internal model can be built using a **physically correct rotation transformation** from the start, completely bypassing the entire "shearing vs. rotation" dilemma.

---
### **Practical Challenges**

While powerful, this method has two significant practical challenges:

1.  **Computational Complexity**: Solving a large, coupled nonlinear system is far more computationally intensive than the simple linear algebra of the two-stage method.
2.  **The Need for a Good Initial Guess**: Nonlinear solvers need a starting point. If the initial guesses for the parameters are too far from the true values, the solver might fail to converge or get trapped in a "local minimum"—a solution that looks optimal but isn't the true best fit.

This leads to a crucial and elegant point: **the best way to get a good initial guess for the advanced nonlinear method is to run the simple two-stage method first!**

The decoupled method provides excellent estimates for the tilt angles and form coefficients, which can then be fed into the Gauss-Newton loop as a high-quality starting point. This ensures a fast and reliable convergence to the true global minimum. 🤓

---
### **3. The Complete Analysis Workflow**
The solution is found using a sophisticated, multi-stage process that ensures both robustness and accuracy.

#### **Stage 1: Generate a Robust Initial Guess**
Because nonlinear solvers can be sensitive, we first run a simpler, two-stage analysis to get a high-quality starting point.
1.  **Separate Form and Tilt**: A best-fit line ( $y = mx + b$ ) is calculated for each of the four raw measurement channels. The slope ( $m$ ) gives a preliminary estimate of the combined tilt, and the residuals ( $I(x) - (mx + b)$ ) give a preliminary estimate of the combined form.
2.  **Solve for Initial Parameters**: Simple algebraic manipulation of the preliminary tilts and forms yields an initial guess for $\alpha, \beta, \gamma$, and the coefficients that describe the form profiles (typically using polynomials).

#### **Stage 2: The Nonlinear Least-Squares Optimization**
This is the core of the method, where all parameters are solved for simultaneously.
1.  **The Objective Function**: A function is defined that takes a vector of trial parameters (all tilts and form coefficients) and builds a full, physically correct model of the four measurement channels based on the equations above. This model includes:
    * **Rigid Body Rotation**: It applies a true rotational transformation to remove tilt, which is more physically accurate than simple subtraction.
    * **Artefact Thickness Correction**: It accounts for the lateral shift between the top and bottom surfaces caused by the artefact's thickness, making the model more rigorous.
2.  **The Optimization Loop**: The `scipy.optimize.least_squares` solver is initiated with the **initial guess** from Stage 1. It then iteratively adjusts all parameters, calls the objective function to see how well the new model fits the real data, and continues this process until it finds the single combination of parameters that minimizes the total error (the sum of squared differences) across all four channels.
3.  **How Tilts Are Optimized**: The solver doesn't optimize the tilts in isolation. It finds the values for $\alpha, \beta,$ and $\gamma$ that, when used in the rotation model, allow the form profiles to best explain the measured data. Their "correctness" is judged by their contribution to the entire system's fit.

#### **Stage 3: Final Results**
Once the solver converges, the final, optimized parameter vector is unpacked. This provides the definitive values for the three tilt angles and the coefficients for the three form profiles, which represent the complete solution.

---
### **4. Sources of Error and Areas for Improvement**

The accuracy of this method is limited by several factors that must be understood.

* **Systematic Errors**: The largest source of uncertainty is often an **imperfect mechanical reversal**. Any unintended shift during the 180° rotation introduces errors that cannot be solved for and must be minimized through careful experimental design and included in the final uncertainty budget.
* **Random Errors**: Electronic noise, vibration, and thermal drift introduce random errors into the measurements. The effect of this noise is significantly reduced by the **holistic nature of the least-squares fit**, which finds a best-fit across all four noisy channels simultaneously.
* **Modeling Errors**:
    * **Shearing vs. Rotation**: The choice of how to remove tilt introduces a small modeling error. The error from the shearing approximation is systematic and predictable ( $Error \propto \theta^2$ ), while the error from rotation is a data-processing error from interpolation ( $Error \propto (\text{local curvature})$ ). A statistical (RMS) comparison of these errors can determine the best method for a given dataset.
    * **Parameterization**: The form profiles are modeled using a basis function (e.g., polynomials, splines, or Fourier series). The choice of the **order** of this function is critical. Too low an order will not be able to capture the true shape of the surface (underfitting), while too high an order can lead to fitting noise instead of the signal (overfitting). This choice must be made carefully based on the expected nature of the surfaces.
