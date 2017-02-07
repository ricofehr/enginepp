__kernel
void collision_step1(__global float *box1, __global float *box2, pipe __write_only float *distances) {

    int fact = get_global_id(0);

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
    if ((x2 < x1 + w1)    /* Too much at right */
        && (x2 + w2 > x1) /* Too much at left */
        && (y2 + h2 < y1) /* Too much at top */
        && (y2 > y1 + h1) /* Too much at bottom */
        && (z2 > z1 + d1) /* Too much at back */
        && (z2 + d2 < z1)) { /* Too much at front */
        write_pipe(distances, 0.1 * fact);
    }
}

__kernel
void collision_step2(pipe __read_only float *distancesin, __global float *box1, __global float *box2, __global float *distancesout) {

    int fact = get_global_id(0);

    float distin = 1.0f;
    float x1, y1, z1, w1, h1, d1, move1x, move1y, move1z;
    float x2, y2, z2, w2, h2, d2, move2x, move2y, move2z;

    while (read_pipe(distancesin, &distin));

    if (distin == 1.0f)
        return;

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
    x1 += (distin + fact * 0.001) * move1x;
    y1 += (distin + fact * 0.001) * move1y;
    z1 += (distin + fact * 0.001) * move1z;
    x2 += (distin + fact * 0.001) * move2x;
    y2 += (distin + fact * 0.001) * move2y;
    z2 += (distin + fact * 0.001) * move2z;
    if ((x2 >= x1 + w1)
            || (x2 + w2 <= x1)
            || (y2 + h2 >= y1)
            || (y2 <= y1 + h1)
            || (z2 <= z1 + d1)
            || (z2 + d2 >= z1)) {
        distancesout[fact] = distin + 0.001 * (fact - 1);
    } else {
        distancesout[fact] = distin + 0.001 * (fact - 1);
    }
}

