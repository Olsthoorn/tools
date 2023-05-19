        for il in [0, 1]:
            sigma = 4 * np.pi * kD / kw['Q'] * PhiI[:, il, 0]
            if rho in [0.01, 1.0, 1.5, 3.]:
                labelN = 'num: r/B = {:.4g}'.format(rho)
                labelA = 'ana: r/B = {:.4g}'.format(rho)
                lw = 2.
            else:
                labelN = '_'
                labelA = '_'
                color = 'k'
                lw = 0.5
            ax.plot(tauA[1:], sigma[1:], '-', color=color, lw=lw, label=labelN)
            ax.plot(tauA[1:], sa[1:,il], 'x', color=color, lw=lw, label=labelA)
