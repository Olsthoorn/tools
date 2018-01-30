$STORAGE:2

c........Double precision I1(X)

      double precision function di1bes (xx)
c
      implicit double precision (a-h, o-z)
      double precision xx, sqrtx, ar1(10), br, br1, br2, zz, z2, numrtr
     .                ,dnmntr, ar2(17)
      integer*2 ii
      data ar1/.14940 52814 740  d1, -.36202 64202 42263d3,              1  2
     .         .22054 97222 60336d5, -.40892 80849 44275d6,              3  4
     .         .10277 66923 71524d7,                                     5
     .        1.0                d0, -.63100 32005 51590d3,              6  7
     .         .49681 19495 33398d5, -.10042 54281 33695d7,              8  9
     .         .26028 87678 9105 d7/                                    10
      data ar2/.20717 57672 32792d-26,  .25709 19055 84414d-23,          1  2
     .         .30627 92836 56135d-20,  .26137 27721 58124d-17,          3  4
     .         .17846 93614 10091d-14,  .96362 88915 18450d-12,          5  6
     .         .41006 89068 47159d-09,  .13545 62288 41096d-06,          7  8
     .         .33947 28903 08516d-04,  .62472 61951 27003d-02,          9 10
     .         .80614 48788 21295d-00,  .68210 05679 80207d+02,         11 12
     .         .34106 97522 84422d+04,  .84070 57728 77836d+05,         13 14
     .         .68146 79652 62502d+06, -.22258 36740 00860d+04,         15 16
     .         .13629 35930 52499d+07/                                  17
c
      if (abs(xx) .lt. 1.0e-30) then
         di1bes = 0.0d0
      else if (abs(xx) .le. 15.0) then
         zz = xx * xx
         dnmntr = zz * (zz + ar2(16)) + ar2(17)
         numrtr = zz * ar2(1) + ar2(2)
         do 10 ii=3, 15
            numrtr = zz * numrtr + ar2(ii)
   10    continue
         di1bes = xx * (numrtr / dnmntr)
      else
         xx = dabs(xx)
         sqrtx = dsqrt(xx)
         br1 = 0.0d0
         br2 = 0.0d0
         zz = 30.0d0 / xx - 1.0d0
         z2 = zz + zz
         do 30 ii=1, 4
            br = z2 * br1 - br2 + ar1(ii)
            br2 = br1
            br1 = br
   30    continue
         numrtr = zz * br1 - br2 + ar1(5)
         br1 = 0.0d0
         br2 = 0.0d0
         do 40 ii=6, 9
            br = z2 * br1 - br2 + ar1(ii)
            br2 = br1
            br1 = br
   40    continue
         dnmntr = zz * br1 - br2 + ar1(10)
         di1bes = dexp(xx) * (numrtr / dnmntr) / sqrtx
      end if
c
***   di1bes = di1bes * dexp(-xx)
c
      return
      end
