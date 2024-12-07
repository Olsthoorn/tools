def gw2rd(GWxy, verbose=False):
    """Return RD coordinates given GW coordinates.

    RD is then Dutch national triangulized coordinates system.
    GW is Gemeentewaterleidingen (Amsterdam Water Supply Dunes system, which is perpendiular
        to the coast line along the Amstedam Water Supply Dunes with zero x at the beach, i.e.
        the distance from the coast.

    Parameters
    ----------
    GWxy: array (n, 2) of floats
        GW coofdinates

    Returns
    -------
    RDxy: np.array (n, 2) of floats
        RD coordinates

    >>>gw2rd(np.array([[61892.94, 885.364]]))
    array([[155000, 463000]])

    See also
    --------
    rd2gw wgs2rd rd2wgs

    @ Pierre Kamps 1999-11-10,
    @ TO 2024-02-10
    """
    assert GWxy.shape[1] == 2, 'RDxy must have shape(n, 2), not {}'.format(shape(RDxy))

    GWaugT = np.ones((3, GWxy.shape[0]))
    GWaugT[:2, :] = GWxy.T

    angle = 22.58378 * np.pi / 180
    xca, yca  =  155000.0, 463000.0
    xrn, yrn  =  -57486.9,  22951.5

    dx, dy = xrn + xca, yrn + yca

    M = np.array([[ np.cos(angle), np.sin(angle), dx],
                  [-np.sin(angle), np.cos(angle), dy],
                  [             0,             0,  1]])
    if verbose:
        print('M = \n', M)

    RDxy = (M @ GWaugT).T[:, :2]

    return RDxy


def rd2gw(RDxy, verbose=False):
    """Return GW coordinates given RD coordinates.

    RD is then Dutch national triangulized coordinates system.
    GW is Gemeentewaterleidingen (Amsterdam Water Supply Dunes system, which is perpendiular
        to the coast line along the Amstedam Water Supply Dunes with zero x at the beach, i.e.
        the distance from the coast.

    Parameters
    ----------
    RDxy: array (n, 2) of floats
        RD coofdinates

    Returns
    -------
    GWxy: np.array (n, 2) of floats
        GW coordinates


    >>>rd2gw(np.array([[155000, 463000]]))
    np.array([[61892.94, 885.364]])

    See also
    --------
    gw2rd wgs2rd rd2wgs

    @ Pierre Kamps 1999-11-10,
    @ TO 2024-02-10
    """
    assert RDxy.shape[1] == 2, 'RDxy must have shape(n, 2), not {}'.format(shape(RDxy))

    RDaugT = np.ones((3, RDxy.shape[0]))
    RDaugT[:2, :] = RDxy.T

    angle = 22.58378 * np.pi / 180
    xca, yca  =  155000.0, 463000.0
    xrn, yrn  =  -57486.9,  22951.5

    dx, dy = xrn + xca, yrn + yca

    M = np.array([[ np.cos(angle), np.sin(angle), dx],
                  [-np.sin(angle), np.cos(angle), dy],
                  [             0,             0,  1]])
    if verbose:
        print('np.linalg.inv(M) = \n', np.linalg.inv(M))

    GWxy = (np.linalg.inv(M) @ RDaugT).T[:, :2]

    return GWxy

if __name__ == '__main__':

 print(Testing the conversions:')
  gw = np.array([[ 61892.94,    885.364]])
  print(gw2grd(gw))
  print(rd2gw(gw2rd(gw)))

  rd = np.array([[155000.04, 463000.000]])
  print(rd2gw(rd))
  print(gw2rd(rd2gw(rd)))
