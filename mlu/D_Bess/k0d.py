$STORAGE:2

c.....Computation of K0 in double precision

      double precision function dk0bes (xx)
c
      implicit double precision (a-h, o-z)
      include 'CBBESS'
      integer ii
c
*     if (x .lt. 1.0d-40)then
*        dk0bes = 1.0d40
*     else if (xx .lt. 1.d-37) then
*        dk0bes = -dlog(xx)
*     else if (xx .gt. 200.d0) then
*        dk0bes = 0.0d0
*     else if (xx .lt. 1.5d0) then
      if (xx .lt. 1.5d0) then
c...........NUMAL 35173; x < 1,5
         xx = dmax1(xx, 1.d-40)
         sum0 = dlog(2.d0 / xx) - .57721566490153286d0
         dd = sum0
         rr = 1.d0
         term = 1.d0
         tt = (xx * xx) / 4.d0
         kk = 0.d0
   10    continue
            kk = kk + 1.d0
            term = term * tt * rr * rr
            dd = dd + rr
            rr = 1.d0 / (kk + 1.d0)
            t0 = term * dd
            sum0 = sum0 + t0
            sumsum = dabs(t0 / sum0)
         if (sumsum .gt. 1.d-15) goto 10
         dk0bes = sum0

      else if (xx .gt. 5.0d0) then
c...........NUMAL 35178 deel 2; x > 5,0
         br1 = 0.d0
         br2 = 0.d0
         y1 = 1.d1 / xx - 1.d0
         y2 = 2.d0 * y1
         do 20 ii=1, 14
            brtmp = y2 * br1 - br2 + fac2(ii)
            br2 = br1
            br1 = brtmp
   20    continue
         f0 = y1 * br1 - br2 + .9884081742308258d0
         tmprt = dsqrt(1.5707963267949d0 / xx)
         dk0bes = f0 * tmprt * dexp(-xx)

      else
c...........NUMAL 35178 deel 1; 1,5 < x < 5,0
         xxb20 = xx * 2.d1
         rj = 1.0d0
         sum1 = .5d0
         do 30 ii=1, 20
            rjsqr = rj * rj
            sqrtex = dsqrt(1.d0 + rjsqr / xxb20)
            sum1 = sum1 + fac3(ii) / sqrtex
            rj = rj + 1.d0
   30    continue
         dk0bes = dexp(-xx) * sum1 / dsqrt(5.d0 * xx)
      end if
c
      return
      end
