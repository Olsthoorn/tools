$STORAGE:2
c.....Computation of K1 in double precision

      double precision function dk1bes (xx)
c
      implicit double precision (a-h, o-z)
      include 'CBBESS'
      integer ii
c
*     if (xx .lt. 1.0d-40)then
*        dk1bes = 1.0d40
*     else if (xx .lt. 1.d-37) then
      if (xx .lt. 1.d-37) then
         dk1bes = 1.0d0 / xx
*     else if (xx .gt. 200.d0) then
*        dk1bes = 0.0d0
      else if (xx .lt. 1.5d0) then
*     if (xx .lt. 1.5d0) then
c...........NUMAL 35173
         xx = dmax1(xx, 1.d-40)
         som0 = dlog(2.d0 / xx) - .57721566490153286d0
         dd = som0
         sum1 = -1.d0 - 2.d0 * dd
         cc = sum1
         rr = 1.d0
         term = 1.d0
         tt = (xx * xx) / 4.d0
         kk = 0.d0
   10    continue
            kk = kk + 1
            term = term * tt * rr * rr
            dd = dd + rr
            cc = cc - rr
            rr = 1.d0 / (kk + 1.d0)
            cc = cc - rr
            t0 = term * dd
            t1 = term * cc * rr
            sum0 = sum0 + t0
            sum1 = sum1 + t1
            sumsum = dabs(t1 / sum1)
         if (sumsum .gt. 1.d-15) goto 10
         dk1bes = (sum1 * tt + 1.d0) / xx

      else if (xx .gt. 5.0d0) then
c...........NUMAL 35178 deel 2
         er = 0.d0
         erp1 = 0.d0
         br1 = 0.d0
         br2 = 0.d0
         cr1 = 0.d0
         cr2 = 0.d0
         rr = 3.d1
         y1 = 1.d1 / xx - 1.d0
         y2 = 2.d0 * y1
         do 20 ii=1, 14
            rr = rr - 2.d0
            brtmp = y2 * br1 - br2 + fac2(ii)
            crtmp = y2 * cr1 - cr2 + er
            erm1 = rr * fac2(ii) + erp1
            erp1 = er
            er = erm1
            br2 = br1
            br1 = brtmp
            cr2 = cr1
            cr1 = crtmp
   20    continue
         f0 = y1 * br1 - br2 + .9884081742308258d0
         f1 = y1 * cr1 - cr2 + er / 2.d0
         expx = dsqrt(1.5707963267949d0 / xx)
         f0 = f0 * expx
         dk1bes = (1.d0 + .5d0 / xx) * f0 +
     .    (1.d1 / xx / xx) * expx * f1
         dk1bes = dk1bes * dexp(-xx)

      else
c...........NUMAL 35178 deel 1

         xxb20 = xx * 2.d1
         rj = 1.0d0
         sum2 = 0.d0
         do 30 ii=1, 20
            rjsqr = rj * rj
            sqrtex = dsqrt(1.d0 + rjsqr / xxb20)
            sum2 = sum2 + 0.1d0 * rjsqr * fac3(ii) * sqrtex
            rj = rj + 1.d0
   30    continue
         dk1bes = 2.0 * dexp(-xx) * sum2 / dsqrt(5.d0 * xx)
      end if

c
      return
      end
