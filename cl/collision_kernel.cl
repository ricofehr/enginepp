__kernel
void collision(__global float *box1, __global float *box2, __global float *distances) {
    int fact = get_global_id(0);
    float granularity = get_global_size(0);
    float x1, y1, z1, w1, h1, d1, move1x, move1y, move1z;
    float x2, y2, z2, w2, h2, d2, move2x, move2y, move2z;

    x1 = box1[0];
    y1 = box1[1];
    z1 = box1[2];
    w1 = box1[3];
    h1 = box1[4];
    d1 = box1[5];
    move1x = box1[6] / granularity;
    move1y = box1[7] / granularity;
    move1z = box1[8] / granularity;
    x2 = box2[0];
    y2 = box2[1];
    z2 = box2[2];
    w2 = box2[3];
    h2 = box2[4];
    d2 = box2[5];
    move2x = box2[6] / granularity;
    move2y = box2[7] / granularity;
    move2z = box2[8] / granularity;

    x1 += (fact + 1) * move1x;
    y1 += (fact + 1) * move1y;
    z1 += (fact + 1) * move1z;
    x2 += (fact + 1) * move2x;
    y2 += (fact + 1) * move2y;
    z2 += (fact + 1) * move2z;

    if (x2 < x1 + w1 && x2 + w2 > x1 && y2 + h2 < y1 &&
        y2 > y1 + h1 && z2 > z1 + d1 && z2 + d2 < z1)
        distances[fact] = fact / granularity;
}

