package com.company;

import edu.princeton.cs.algs4.StdOut;

public class ForTemp {
    public static void main(String[] args) {
        int LengthOfArray = 100;
        double a[] = new double[LengthOfArray];
        for (int i = 0; i < LengthOfArray; i++) {
            a[i] = i + 1;
        }
        double m = 0.0;
        for (int i = 0; i < LengthOfArray; i++) {
            if (a[i] > m) {
                m = a[i];
                StdOut.println("m changed ");
            }
        }
        StdOut.println(m);
    }
}
