
def dk1bes:
'''
Computation of K1 in double precision
double precision function dk1bes (xx)
implicit double precision (a-h, o-z)
      integer ii
'''
from CBBESS import fac2, fac3

      if xx < 1.e-37:
         dk1bes = 1.0 / xx
      elif xx < 1.5:
         # NUMAL 35173
         xx = max1(xx, 1.e-40)
         som0 = log(2.0 / xx) - .57721566490153286d0
         dd = som0
         sum1 = -1.d0 - 2.0 * dd
         cc = sum1
         rr = 1.0
         term = 1.0
         tt = (xx * xx) / 4.0
         kk = 0.0
         while True:
            kk = kk + 1
            term = term * tt * rr * rr
            dd = dd + rr
            cc = cc - rr
            rr = 1.0 / (kk + 1.0)
            cc = cc - rr
            t0 = term * dd
            t1 = term * cc * rr
            sum0 += t0
            sum1 += t1
            sumsum = abs(t1 / sum1)
             if sumsum <= 1.e-15:
                break
         dk1bes = (sum1 * tt + 1.d0) / xx

      elif xx > 5.0:
         #.NUMAL 35178 deel 2
         er = 0.0
         erp1 = 0.0
         br1 = 0.0
         br2 = 0.0
         cr1 = 0.0
         cr2 = 0.0
         rr = 3.e1
         y1 = 1.e1 / xx - 1.0
         y2 = 2.d0 * y1
         for ii in range(1,15):
            rr = rr - 2.0
            brtmp = y2 * br1 - br2 + fac2(ii)
            crtmp = y2 * cr1 - cr2 + er
            erm1 = rr  * fac2(ii) + erp1
            erp1 = er
            er = erm1
            br2 = br1
            br1 = brtmp
            cr2 = cr1
            cr1 = crtmp

         f0 = y1 * br1 - br2 + 0.9884081742308258d0
         f1 = y1 * cr1 - cr2 + er / 2.0
         expx = np.sqrt(1.5707963267949d0 / xx)
         f0 = f0 * expx
         dk1bes = (1.0 + .50 / xx) * f0 +
     .    (1.e1 / xx / xx) * expx * f1
         dk1bes = dk1bes * dexp(-xx)

      else:
         # NUMAL 35178 deel 1
         xxb20 = xx * 2.e1
         rj = 1.0
         sum2 = 0.0
         for ii in range(1, 21):
            rjsqr = rj * rj
            sqrtex = dsqrt(1.0 + rjsqr / xxb20)
            sum2 = sum2 + 0.10 * rjsqr * fac3(ii) * sqrtex
            rj = rj + 1.0
         dk1bes = 2.0 * np.exp(-xx) * sum2 / np.sqrt(5.0 * xx)

      return dk1bes
