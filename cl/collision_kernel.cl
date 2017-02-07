__kernel
void collision(__global float *box1, __global float *box2, __global float *distances) {

    int fact = get_global_id(0);
    int fact2 = 0;

    float x1, y1, z1, w1, h1, d1, move1x, move1y, move1z;
    float x2, y2, z2, w2, h2, d2, move2x, move2y, move2z;

    x1 = box1[0];
    y1 = box1[1];
    z1 = box1[2];
    w1 = box1[3];
    h1 = box1[4];
    d1 = box1[5];
    move1x = box1[6];
    move1y = box1[7];
    move1z = box1[8];
    x2 = box2[0];
    y2 = box2[1];
    z2 = box2[2];
    w2 = box2[3];
    h2 = box2[4];
    d2 = box2[5];
    move2x = box2[6];
    move2y = box2[7];
    move2z = box2[8];

    /* 1/10 precision */
    x1 += (fact + 1) * 0.1 * move1x;
    y1 += (fact + 1) * 0.1 * move1y;
    z1 += (fact + 1) * 0.1 * move1z;
    x2 += (fact + 1) * 0.1 * move2x;
    y2 += (fact + 1) * 0.1 * move2y;
    z2 += (fact + 1) * 0.1 * move2z;
    if ((x2 >= x1 + w1)    /* Too much at right */
        || (x2 + w2 <= x1) /* Too much at left */
        || (y2 + h2 >= y1) /* Too much at top */
        || (y2 <= y1 + h1) /* Too much at bottom */
        || (z2 <= z1 + d1) /* Too much at back */
        || (z2 + d2 >= z1)) { /* Too much at front */
            distances[fact] = 1.0f; /* Default set 1.0 */
    } else {
        /* Back 1/0 move */
        x1 -= 0.1 * move1x;
        y1 -= 0.1 * move1y;
        z1 -= 0.1 * move1z ;
        x2 -= 0.1 * move2x;
        y2 -= 0.1 * move2y;
        z2 -= 0.1 * move2z;

        /* 1/100 precision */
        fact2 = 1;
        while (1) {
            x1 += 0.001 * move1x;
            y1 += 0.001 * move1y;
            z1 += 0.001 * move1z;
            x2 += 0.001 * move2x;
            y2 += 0.001 * move2y;
            z2 += 0.001 * move2z;
            if ((x2 >= x1 + w1)
                    || (x2 + w2 <= x1)
                    || (y2 + h2 >= y1)
                    || (y2 <= y1 + h1)
                    || (z2 <= z1 + d1)
                    || (z2 + d2 >= z1)) {
                    ++fact2;
                    continue;
            }

            break;
        }
        
        distances[fact] = 0.1 * fact + 0.001 * (fact2 - 1);
    }
}

