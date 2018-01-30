$STORAGE:2

c........Double precision I0(x)

      double precision function di0bes (xx)
c
      implicit double precision (a-h, o-z)
      double precision xx, sqrtx, ar1(10), br, br1, br2, zz, z2, numrtr,
     .                 dnmntr, ar2(18)
      integer*2 ii
      data ar1/.24392 60769 778  d0, -.11559 19781 04435d3,              1  2
     .         .78403 42490 05088d4, -.14346 46313 13583d6,              3  4
     .         .34651 98333 57379d6, 1.00000000000000000d0,              5  6
     .        -.32519 73333 69824d3,  .20312 84361 00794d5,              7  8
     .        -.36184 77792 19653d6,  .86566 52748 32055d6/              9 10
      data ar2/.21058 07228 90567d-22,  .38071 52423 45326d-19,          1  2
     .         .47944 02575 48300d-16,  .43512 59712 62668d-13,          3  4
     .         .30093 11271 12960d-10,  .16022 46793 95361d-7 ,          5  6
     .         .65485 83700 96785d-5 ,  .20259 10841 43397d-2 ,          7  8
     .         .46307 62847 21000d0  ,  .75433 73289 48189d2  ,          9 10
     .         .83079 25418 09429d4  ,  .57166 11305 63785d6  ,         11 12
     .         .21641 55723 61227d8  ,  .35664 44822 44025d9  ,         13 14
     .         .14404 82982 27235d10 , -.30764 69126 82801d4  ,         15 16
     .         .34762 63324 05882d7  , -.14404 82982 27235d10 /         17 18
c
      if (abs(xx) .lt. 1.0e-35) then
         di0bes = 1.0d0
      else if (abs(xx) .le. 15.0) then
         zz = xx * xx
         numrtr = zz * ar2(1) + ar2(2)
         do 10 ii=3, 15
            numrtr = zz *  numrtr + ar2(ii)
   10    continue
         dnmntr = zz + ar2(16)
         do 20 ii=17, 18
            dnmntr = zz *  dnmntr + ar2(ii)
   20    continue
         di0bes = -1.0 * (numrtr / dnmntr)
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
         br2 = 0.0d0
         br1 = 0.0d0
         zz = 30.0d0 / xx - 1.0d0
         z2 = zz + zz
         do 40 ii=6, 9
            br = z2 * br1 - br2 + ar1(ii)
            br2 = br1
            br1 = br
   40    continue
         dnmntr = zz * br1 - br2 + ar1(10)
         di0bes = dexp(xx) * (numrtr / dnmntr) / sqrtx
      end if
***   di0bes = di0bes * dexp(-xx)
c
      return
      end
