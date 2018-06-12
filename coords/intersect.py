#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 13:44:00 2017

@author: Theo
"""
import numpy as np
import matplotlib.pyplot as plt

AND = np.logical_and
OR  = np.logical_or
NOT = np.logical_not

def perp(xp, yp, x0, y0, alpha):
    '''returns intersection line through point perpendicular to given line
    parameters
    ----------
    xp, yp : floats
        coordinate of point from which to draw line perpendicular to given line
    x0, y0, alpah : floats
        given line, x0, y0 point of, and alpha angle with respect to east (degrees)

    TO 171127
    '''

    ex = np.cos(np.pi * alpha / 180.)
    ey = np.sin(np.pi * alpha / 180.)

    M = np.array([[ex, ey], [ey, -ex]])
    Mm1 = np.linalg.inv(M)

    print('Mm1=\n', Mm1, '\n')

    lm = np.dot(Mm1, np.array([[xp - x0], [yp - y0]]))

    print('lm = \n', lm, '\n')

    lam = lm[0]
    mu  = lm[1]
    print('lam = ', lam)
    print('mu  =', mu)

    xa = x0 + lam * ex
    ya = y0 + lam * ey

    xb = xp + mu * -ey
    yb = yp + mu * ex

    plt.plot(x0, y0, 'bo')
    plt.plot(xp, yp, 'ro')
    plt.plot([x0, xa], [y0, ya], 'b')
    plt.plot([xp, xb], [yp, yb], 'r')
    plt.plot(xa, ya, 'bx')
    plt.plot(xb, yb, 'r+')

    plt.text(x0, y0, '(x0, y0)', ha='left')
    plt.text(xp, yp, '(xp, yp)', ha='left')
    plt.text(xa, ya, 'Intersection', ha='left')
    plt.text(xb, yb, 'Intersection', ha='left')
    plt.show()

#x0, y0, alpha = 0., -143., 16.
#xp, yp = 323., 521.

#perp(xp, yp, x0, y0, alpha)


def perpMany(Xp, Yp, x0, y0, alpha, verbose=False):
    '''Return distance to line given by x0, y0, alpha for many points given by Xp, Yp
    
    parameters
    ----------    
    Xp : np.ndarray of floats
        x coordinates of points
    Yp : np.ndarra of floats
        y coordinate of points
    x0, y0, alpha : 3 floats
        x0, y0, angle(degrees) of point defining the line.
    '''

    ex = np.cos(np.pi / 180. * alpha)
    ey = np.sin(np.pi / 180. * alpha)

    Mi = np.linalg.inv(np.array([[ex, ey], [ey, -ex]]))

    lammu = np.dot(Mi, np.array([Xp - x0, Yp - y0]))

    lam = lammu[0]
    mu  = lammu[1]

    if np.all(np.isnan(mu)):
        return mu
    
    if verbose:
        fig, ax = plt.subplots()
        
        Xa = x0 + lam * ex
        Ya = y0 + lam * ey
    
        Xb = Xp + mu * -ey
        Yb = Yp + mu *  ex
    
        ax.title('Testing perpendicular lines to n points')
        ax.set_xlabel('x')
        ax.set_ylabel('x')
        ax.set_xlim((-60, 60))
        ax.set_ylim((-60, 60))
        ax.text(x0, y0, '(x0, y0)', ha='left')
    
        ax.plot(x0, y0, 'r.')
    
        ax.plot([Xp, Xb], [Yp, Yb], 'b-')
        ax.plot( Xp,       Yp,      'bo')
        ax.plot( Xb,       Yb,      'bx')
        ax.plot( Xa,       Ya,      'r+')
    
        X0 = np.zeros_like(Xa) + x0
        Y0 = np.zeros_like(Ya) + y0
    
        ax.plot([X0, Xa], [Y0, Ya], 'r-')
    
        for xpp, ypp, x, y, m in zip(Xp, Yp, Xb, Yb, mu):
            xm = 0.5 * (xpp + x)
            ym = 0.5 * (ypp + y)
            ax.text(xm, ym, 'mu={:.3g}'.format(m), ha='left')
    
    return mu



def ln2alpha(line):
    '''returns lines as (x0, y0, alpha) when given (x0, y0, x1, y1)
    paramters
    ---------
    line : tuple of 3 or 4 floats
        line = (x0, y0, x1, y1) or line= (x0, y0, alpha)
        with alpha in degrees
    returns
    -------
    line : tuple
        line = (x0, y0, alpha)
        alpha in degrees
    '''
    if len(line) == 3:
        return line
    else:
        if len(line) != 4:
            raise ValueError("len(line) must be 3 or 4")
        x0, y0, x1, y1 = line
        if x0 > y0: # orient line northward
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        alpha = np.arctan2(y1 - y0, x1 - x0) * 180. / np.pi
        return x0, y0, alpha


def dist2line(Xp, Yp, x0, y0, alpha, verbose=False):
    '''Return and possibly plot distance from Xp, Yp to  line.

    parameters
    ----------
    Xp, Yp : np.ndarrays
        x and y coordinates of points
    x0, y0, alpha : float, float, float
        point on line and angle in degrees with respect to east
    verbose: bool
        if true then plot points, line and perpendicular lines
    returns
    -------

    mu : np.ndarray
        distance of bores to line

    TO 171127
    '''

    Xp = np.array(Xp)
    Yp = np.array(Yp)

    ex = np.cos(np.pi / 180. * alpha)
    ey = np.sin(np.pi / 180. * alpha)

    Mi = np.linalg.inv(np.array([[ex, ey], [ey, -ex]]))

    lammu = np.dot(Mi, np.array([Xp - x0, Yp - y0]))

    lam = lammu[0]
    mu  = lammu[1]

    if not verbose:
        return mu

    Xa = x0 + lam * ex
    Ya = y0 + lam * ey

    Xb = Xp + mu * -ey
    Yb = Yp + mu *  ex

    if np.all(np.isnan(mu)):
        return mu

    if verbose:
        X0 = np.zeros_like(Xa) + x0
        Y0 = np.zeros_like(Ya) + y0
    
        xmin = min((np.nanmin(Xa), np.nanmin(Xp), np.nanmin(X0)))
        xmax = max((np.nanmax(Xa), np.nanmax(Xp), np.nanmax(X0)))
        ymin = min((np.nanmin(Ya), np.nanmin(Yp), np.nanmin(Y0)))
        ymax = max((np.nanmax(Ya), np.nanmax(Yp), np.nanmax(Y0)))

        fig, ax = plt.subplots()
        ax.set_title('Testing perpendicular lines to n points')
        ax.set_xlabel('x')
        ax.set_ylabel('x')

        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
        ax.text(x0, y0, '(x0, y0)', ha='left')
    
        ax.plot(x0, y0, 'r.')
    
        ax.plot([Xp, Xb], [Yp, Yb], 'b-')
        ax.plot( Xp,       Yp,      'bo')
        ax.plot( Xb,       Yb,      'bx')
        ax.plot( Xa,       Ya,      'r+')
    
    
        ax.plot([X0, Xa], [Y0, Ya], 'r-')

        for xpp, ypp, x, y, m in zip(Xp, Yp, Xb, Yb, mu):
            xm = 0.5 * (xpp + x)
            ym = 0.5 * (ypp + y)
            ax.text(xm, ym, 'mu={:.3g}'.format(m), ha='left')

    return mu


def dist2polyline(points, polyline, alpha, verbose=False, maxdist=None):
    '''return distances from points to polyline given angle alpha
    parameters
    ----------
    points : tuple (Xp, Yp) or sequence of tuples (Xpi, Ypi) or np.ndarray
        the coordinates from which the distance to line is tob computed.
    polyle : like points, but interpreted as a polyline
        line to which the distance is to be computed.
    alpha : float
        angle in degrees with respect to east under whicht the lines
        through the points will intersect the polyline
    verbose: bool
        if true then plot points, line and perpendicular lines
    returns
    -------
    mu : np.ndarray
        distance of bores to line

    TO 171127
    '''

    if isinstance(maxdist, (int, float)):
        maxdist = (-maxdist, maxdist)

    points = np.array(points)
    Xp = points[0]
    Yp = points[1]
    line = np.array(polyline)
    X0 = line[:, 0]
    Y0 = line[:, 1]
    Dx = np.diff(X0)
    Dy = np.diff(Y0)

    ex = np.cos(np.pi / 180. * alpha)
    ey = np.sin(np.pi / 180. * alpha)

    Mu = np.zeros_like(Xp) * np.nan

    for i, (dx, dy, x0, y0) in enumerate(zip(Dx, Dy, X0[:-1], Y0[:-1])):

        Mi = np.linalg.inv(np.array([[dx, -ex], [dy, -ey]]))

        lammu = np.dot(Mi, np.array([Xp - x0, Yp - y0]))

        lam = lammu[0]
        mu  = lammu[1]

        Mu[AND(lam>=0, lam<=1)] = mu[AND(lam>=0, lam<=1)]

    if maxdist is not None:
        Mu[OR(Mu<maxdist[0], Mu>maxdist[1])] = np.nan


    if np.all(np.isnan(Mu)):
        raise Warning("All distances are Nan, Can't plot the points")
        return Mu

    if verbose:

        Xa = Xp + Mu * ex
        Ya = y0 + Mu * ey
    
        xmin = min((np.nanmin(Xa), np.nanmin(Xp), np.nanmin(X0)))
        xmax = max((np.nanmax(Xa), np.nanmax(Xp), np.nanmax(X0)))
        ymin = min((np.nanmin(Ya), np.nanmin(Yp), np.nanmin(Y0)))
        ymax = max((np.nanmax(Ya), np.nanmax(Yp), np.nanmax(Y0)))

        fig, ax = plt.subplots()
        ax.set_title('Testing distance to polyline')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
        ax.plot(X0, Y0, 'b', linewidth=3, label='polyline')
        ax.plot(Xp, Yp, 'r.')

        for xp, yp, mu in zip(Xp, Yp, Mu):
            if not np.isnan(mu):
                ax.plot([xp, xp + ex * mu], [yp, yp + ey * mu], 'r')
    
        ax.legend(loc='best')

    return Mu


if __name__ == '__main__' :

    n = 10
    Xp = (np.random.random(n) - 0.5) * 100.
    Yp = (np.random.random(n) - 0.5) * 100.
    x0, y0, alpha = 45., 45., 0.35
    
    perpMany(Xp, Yp, x0, y0, alpha, verbose=True)

