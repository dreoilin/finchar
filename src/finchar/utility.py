import numpy as np

from .numerical import interp1


def EKV_param_extraction(lk, mode, **kwargs):
    return XTRACT(lk, mode, **kwargs)

def XTRACT(lk, mode, **kwargs):
    """
    EKV param extraction algorithm.

    Two modes of operation:
    1)  L, VSB are scalars. VDS scalar or column vector.
        rho is an optional scalar

    2)
    """
    plot = kwargs.get('plot', False)

    if mode == 1:
        L   =   kwargs.get('L', min(lk['L']))
        VDS =   kwargs.get('VDS', lk['VDS'])
        VSB =   kwargs.get('VSB', 0.0)
        rho =   kwargs.get('rho', 0.6)
        UDS =   kwargs.get('UDS', np.arange(0.025, 1.2+0.025, 0.025))

        UT  =  ( 0.0259 * lk['TEMP']/300 ).item()

        # find n(UDS)
        gm_ID = lk.look_up('GM_ID', VDS=UDS.T, VSB=VSB, L=L)
        # get max value from each column
        M = np.amax(gm_ID.T, axis=1)
        nn = 1/(M*UT)
        # find VT(UDS)
        q = 1/rho -1
        i = q**2 + q
        VP = UT * (2 * (q-1) + np.log(q))
        gm_IDref = rho * M
        # have to use linear interpolation here. gm_ID is not monotonic
        VGS = [float(interp1(gm_ID[:,k], lk['VGS'], kind='pchip')(gm_IDref[k])) for k in range(len(UDS))]
        
        Vth = VGS - nn*VP
        #find JS(UDS) ===============
        Js = lk.lookup('ID_W',GM_ID=gm_IDref, VDS=UDS, VSB=VSB, L=L).diagonal()/i 
        
        # DERIVATIVES ===============
        UDS1 = .5*(UDS[:-1] + UDS[1:])
        UDS2 = .5*(UDS1[:-1] + UDS1[1:])
        
        diffUDS = np.diff(UDS)
        diffUDS1 = np.diff(UDS1)

        # subthreshold slope ============
        diff1n = np.diff(nn)/diffUDS
        diff2n = np.diff(diff1n)/diffUDS1

        # threshold voltage =============
        diff1Vth = np.diff(Vth)/diffUDS
        diff2Vth = np.diff(diff1Vth)/diffUDS1

        # log specific current ============
        diff1logJs = np.diff(np.log(Js))/diffUDS
        diff2logJs = np.diff(diff1logJs)/diffUDS1

        # n(VDS), VT(VDS) , JS(VDS) ===========
        n  = interp1(UDS, nn, kind='pchip')(VDS) 
        VT = interp1(UDS, Vth, kind='pchip')(VDS)
        JS = interp1(UDS, Js, kind='pchip')(VDS)

        d1n = interp1(UDS1, diff1n, kind='pchip')(VDS)
        d2n = interp1(UDS2, diff2n, kind='pchip')(VDS)

        d1VT = interp1(UDS1, diff1Vth, kind='pchip')(VDS)
        d2VT = interp1(UDS2, diff2Vth, kind='pchip')(VDS)

        d1logJS = interp1(UDS1, diff1logJs, kind='pchip')(VDS)
        d2logJS = interp1(UDS2, diff2logJs, kind='pchip')(VDS)

        if plot:
            #% FIGURE =============
            # setup mpl
            import matplotlib as mpl
            import matplotlib.pyplot as plt
            with mpl.rc_context({'axes.spines.right': False, 
                                 'axes.spines.top': False,
                                 'axes.grid': True}):

                # Figure 1 (Subthreshold slope)
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].plot(UDS, nn, VDS, n, '*')
                ax[0].set_ylabel(r"$n$")
                ax[0].set_xlabel(r"$V_{DS}$ [V]")
                ax[0].set_title("Subthreshold slope ($n$) vs $V_{DS}$")
                ax[1].plot(UDS1, diff1n, VDS, d1n, '*')
                ax[1].set_ylabel(r"$\frac{d(n)}{d(V_{DS})}$")
                ax[1].set_xlabel(r"$V_{DS}$ [V]")
                ax[1].set_title("Derivative of $n$ vs $V_{DS}$")
                ax[2].plot(UDS2, diff2n, VDS, d2n, '*')
                ax[2].set_ylabel(r"$\frac{d^2(n)}{d(V_{DS})^2}$")
                ax[2].set_xlabel(r"$V_{DS}$ [V]")
                ax[2].set_title("Second derivative of $n$ vs $V_{DS}$")
                plt.tight_layout()
                plt.show()

                # Figure 2 (Threshold voltage)
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].plot(UDS, Vth, VDS, VT, '*')
                ax[0].set_ylabel(r"$V_T$ [V]")
                ax[0].set_xlabel(r"$V_{DS}$ [V]")
                ax[0].set_title("Threshold voltage ($V_T$) vs $V_{DS}$")
                ax[1].plot(UDS1, diff1Vth, VDS, d1VT, '*')
                ax[1].set_ylabel(r"$\frac{d(V_T)}{d(V_{DS})}$")
                ax[1].set_xlabel(r"$V_{DS}$ [V]")
                ax[1].set_title("Derivative of $V_T$ vs $V_{DS}$")
                ax[2].plot(UDS2, diff2Vth, VDS, d2VT, '*')
                ax[2].set_ylabel(r"$\frac{d^2(V_T)}{d(V_{DS})^2}$")
                ax[2].set_xlabel(r"$V_{DS}$ [V]")
                ax[2].set_title("Second derivative of $V_T$ vs $V_{DS}$")
                plt.tight_layout()
                plt.show()
                
                # Figure 3 (Specific current density)
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].plot(UDS, Js, VDS, JS, '*')
                ax[0].set_ylabel(r"$J_S$ [A/m$^2$]")
                ax[0].set_xlabel(r"$V_{DS}$ [V]")
                ax[0].set_title("Specific current density ($J_S$) vs $V_{DS}$")
                ax[1].plot(UDS1, diff1logJs, VDS, d1logJS, '*')
                ax[1].set_ylabel(r"$\frac{d(\log(J_S))}{d(V_{DS})}$")
                ax[1].set_xlabel(r"$V_{DS}$ [V]")
                ax[1].set_title(r"Derivative of $\log(J_S)$ vs $V_{DS}$")
                ax[2].plot(UDS2, diff2logJs, VDS, d2logJS, '*')
                ax[2].set_ylabel(r"$\frac{d^2(\log(J_S))}{d(V_{DS})^2}$")
                ax[2].set_xlabel(r"$V_{DS}$ [V]")
                ax[2].set_title(r"Second derivative of $\log(J_S)$ vs $V_{DS}$")
                plt.tight_layout()
                plt.show()
                # plt.figure(1); plt.plot(UDS, Js, VDS, JS, '*')
                # #plt.ylabel(r"$n$"); plt.xlabel(r"$V_{DS}$ [V]")
                # plt.figure(2); plt.plot(UDS1, diff1logJs, VDS, d1logJS, '*')
                # #plt.ylabel(r"$n$"); plt.xlabel(r"$V_{DS}$ [V]")
                # plt.figure(3); plt.plot(UDS2, diff2logJs, VDS, d2logJS, '*')
                # #plt.ylabel(r"$n$"); plt.xlabel(r"$V_{DS}$ [V]")
                # plt.show()

        return (VDS, n, VT, JS, d1n, d1VT, d1logJS, d2n, d2VT, d2logJS) 
    elif mode ==2:
        print("Mode 2 not implemented")
        return
        #VGS =   kwargs.get('VGS', lk['VGS'])
        #ID =   kwargs.get('ID', lk['ID'])
        #rho =   kwargs.get('rho', 0.6)

        #qFo = 1/rho - 1
        #i = qFo * qFo + qFo

        #UT = .026
        #ID = np.atleast_2d(ID)
        #m1, m2 = ID.shape
        #gm_Id = np.diff(np.log(ID))/np.diff(VGS[])
        #z1, b = max(gm_Id)

        # compute VGSo and IDo -------
        #UGS     = .5*(VGS(1:m1-1) + VGS(2:m1));
        #for k = 1:m2,
        #    VGSo(k,1) = interp1(gm_Id(:,k),UGS,z1(k)*rho,'cubic');
        #    IDo(k,1)  = interp1(VGS,ID(:,k),VGSo(k,1),'cubic');
        #end

        #n  = 1./(UT*z1');
        #VT  = VGSo - UT*n.*(2*(qFo-1)+log(qFo));
        #IS = IDo/i;

        #return n, VT, IS 
    else:
        print("Invalid mode")
        return