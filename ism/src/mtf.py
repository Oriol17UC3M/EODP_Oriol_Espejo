from math import pi
from config.ismConfig import ismConfig
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import j1
from numpy.matlib import repmat
from common.io.readMat import writeMat
from common.plot.plotMat2D import plotMat2D
from scipy.interpolate import interp2d
from numpy.fft import fftshift, ifft2
import os

class mtf:
    """
    Class MTF. Collects the analytical modelling of the different contributions
    for the system MTF
    """
    def __init__(self, logger, outdir):
        self.ismConfig = ismConfig()
        self.logger = logger
        self.outdir = outdir

    def system_mtf(self, nlines, ncolumns, D, lambd, focal, pix_size,
                   kLF, wLF, kHF, wHF, defocus, ksmear, kmotion, directory, band):
        """
        System MTF
        :param nlines: Lines of the TOA
        :param ncolumns: Columns of the TOA
        :param D: Telescope diameter [m]
        :param lambd: central wavelength of the band [m]
        :param focal: focal length [m]
        :param pix_size: pixel size in meters [m]
        :param kLF: Empirical coefficient for the aberrations MTF for low-frequency wavefront errors [-]
        :param wLF: RMS of low-frequency wavefront errors [m]
        :param kHF: Empirical coefficient for the aberrations MTF for high-frequency wavefront errors [-]
        :param wHF: RMS of high-frequency wavefront errors [m]
        :param defocus: Defocus coefficient (defocus/(f/N)). 0-2 low defocusing
        :param ksmear: Amplitude of low-frequency component for the motion smear MTF in ALT [pixels]
        :param kmotion: Amplitude of high-frequency component for the motion smear MTF in ALT and ACT
        :param directory: output directory
        :return: mtf
        """

        self.logger.info("Calculation of the System MTF")

        # Calculate the 2D relative frequencies
        self.logger.debug("Calculation of 2D relative frequencies")
        fn2D, fr2D, fnAct, fnAlt = self.freq2d(nlines, ncolumns, D, lambd, focal, pix_size)

        # Diffraction MTF
        self.logger.debug("Calculation of the diffraction MTF")
        Hdiff = self.mtfDiffract(fr2D)

        # Defocus
        Hdefoc = self.mtfDefocus(fr2D, defocus, focal, D)

        # WFE Aberrations
        Hwfe = self.mtfWfeAberrations(fr2D, lambd, kLF, wLF, kHF, wHF)

        # Detector
        Hdet  = self. mtfDetector(fn2D)

        # Smearing MTF
        Hsmear = self.mtfSmearing(fnAlt, ncolumns, ksmear)

        # Motion blur MTF
        Hmotion = self.mtfMotion(fn2D, kmotion)

        # Calculate the System MTF
        self.logger.debug("Calculation of the Sysmtem MTF by multiplying the different contributors")
        Hsys = Hdiff * Hdefoc * Hwfe * Hdet * Hsmear * Hmotion

        # Plot cuts ACT/ALT of the MTF
        self.plotMtf(Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion, Hsys, nlines, ncolumns, fnAct, fnAlt, directory, band)


        return Hsys

    def freq2d(self,nlines, ncolumns, D, lambd, focal, w):
        """
        Calculate the relative frequencies 2D (for the diffraction MTF)
        :param nlines: Lines of the TOA
        :param ncolumns: Columns of the TOA
        :param D: Telescope diameter [m]
        :param lambd: central wavelength of the band [m]
        :param focal: focal length [m]
        :param w: pixel size in meters [m]
        :return fn2D: normalised frequencies 2D (f/(1/w))
        :return fr2D: relative frequencies 2D (f/(1/fc))
        :return fnAct: 1D normalised frequencies 2D ACT (f/(1/w))
        :return fnAlt: 1D normalised frequencies 2D ALT (f/(1/w))
        """
        #TODO
        fstepAlt = 1 / nlines / w
        fstepAct = 1 / ncolumns / w
        eps=1e-10
        fAlt = np.arange(-1 / (2 * w), 1 / (2 * w) - eps, fstepAlt)
        fAct = np.arange(-1 / (2 * w), 1 / (2 * w) - eps, fstepAct)
        fcutoff = D / (lambd*focal)

        #normalize fAct & fAlt
        fnAct = fAct / (1 / w)
        fnAlt = fAlt / (1 / w)
        frAct = fAct / fcutoff
        frAlt = fAlt / fcutoff
        [fnAltxx, fnActxx] = np.meshgrid(fnAlt, fnAct, indexing='ij')  # Please use ‘ij’ indexing or you will get the transpose
        fn2D=np.sqrt(fnAltxx*fnAltxx + fnActxx*fnActxx)
        [fnAltxx, fnActxx] = np.meshgrid(frAlt, frAct,indexing='ij')  # Please use ‘ij’ indexing or you will get the transpose
        fr2D = np.sqrt(fnAltxx * fnAltxx + fnActxx * fnActxx)

        return fn2D, fr2D, fnAct, fnAlt

    def mtfDiffract(self,fr2D):
        """
        Optics Diffraction MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :return: diffraction MTF
        """
        Hdiff = (2 / np.pi) * (np.arccos(fr2D)-fr2D * (1-fr2D**2)**(1/2))
        return Hdiff


    def mtfDefocus(self, fr2D, defocus, focal, D):
        """
        Defocus MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :param defocus: Defocus coefficient (defocus/(f/N)). 0-2 low defocusing
        :param focal: focal length [m]
        :param D: Telescope diameter [m]
        :return: Defocus MTF
        """
        #TODO
        x = np.pi * defocus * fr2D * (1 - fr2D)
        Hdefoc = (2* j1(x)) / x
        return Hdefoc

    def mtfWfeAberrations(self, fr2D, lambd, kLF, wLF, kHF, wHF):
        """
        Wavefront Error Aberrations MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :param lambd: central wavelength of the band [m]
        :param kLF: Empirical coefficient for the aberrations MTF for low-frequency wavefront errors [-]
        :param wLF: RMS of low-frequency wavefront errors [m]
        :param kHF: Empirical coefficient for the aberrations MTF for high-frequency wavefront errors [-]
        :param wHF: RMS of high-frequency wavefront errors [m]
        :return: WFE Aberrations MTF
        """
        #TODO
        Hwfe = np.exp(-fr2D * (1-fr2D) * (kLF*(wLF/lambd)**2 + kHF*(wHF/lambd)**2))
        return Hwfe

    def mtfDetector(self,fn2D):
        """
        Detector MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :return: detector MTF
        """
        #TODO
        Hdet = np.abs(np.sinc(fn2D))
        return Hdet

    def mtfSmearing(self, fnAlt, ncolumns, ksmear):
        """
        Smearing MTF
        :param ncolumns: Size of the image ACT
        :param fnAlt: 1D normalised frequencies 2D ALT (f/(1/w))
        :param ksmear: Amplitude of low-frequency component for the motion smear MTF in ALT [pixels]
        :return: Smearing MTF
        """
        #TODO
        fnAlt = np.repeat(fnAlt[:,None], ncolumns, axis=1)
        Hsmear = np.sinc(ksmear*fnAlt)
        return Hsmear

    def mtfMotion(self, fn2D, kmotion):
        """
        Motion blur MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :param kmotion: Amplitude of high-frequency component for the motion smear MTF in ALT and ACT
        :return: detector MTF
        """
        #TODO
        Hmotion = np.sinc(fn2D * kmotion)
        return Hmotion

    def plotMtf(self,Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion, Hsys, nlines, ncolumns, fnAct, fnAlt, directory, band):
        """
        Plotting the system MTF and all of its contributors
        :param Hdiff: Diffraction MTF
        :param Hdefoc: Defocusing MTF
        :param Hwfe: Wavefront electronics MTF
        :param Hdet: Detector MTF
        :param Hsmear: Smearing MTF
        :param Hmotion: Motion blur MTF
        :param Hsys: System MTF
        :param nlines: Number of lines in the TOA
        :param ncolumns: Number of columns in the TOA
        :param fnAct: normalised frequencies in the ACT direction (f/(1/w))
        :param fnAlt: normalised frequencies in the ALT direction (f/(1/w))
        :param directory: output directory
        :param band: band
        :return: N/A
        """
        #TODO

        # Crear el directorio de salida si no existe
        os.makedirs(directory, exist_ok=True)

        # Índice central ACT (para el corte ALT central)
        centerAct = len(fnAct) // 2

        # Tomamos el corte ALT (vertical)
        Hcuts = {
            "Optics Diffraction MTF": np.abs(Hdiff[:, centerAct]),
            "Defocus MTF": np.abs(Hdefoc[:, centerAct]),
            "Wavefront Error Aberrations MTF": np.abs(Hwfe[:, centerAct]),
            "Detector MTF": np.abs(Hdet[:, centerAct]),
            "Smearing MTF": np.abs(Hsmear[:, centerAct]),
            "Motion blur MTF": np.abs(Hmotion[:, centerAct]),
            "System MTF": np.abs(Hsys[:, centerAct])
        }

        # Tomamos solo frecuencias positivas (de 0 a 0.5)
        mask = fnAlt >= 0
        fnAlt_pos = fnAlt[mask]
        for key in Hcuts:
            Hcuts[key] = Hcuts[key][mask]

        # Crear figura
        plt.figure(figsize=(6, 4))

        # Colores personalizados para mantener consistencia visual
        colors = {
            "Optics Diffraction MTF": "black",
            "Defocus MTF": "red",
            "Wavefront Error Aberrations MTF": "purple",
            "Detector MTF": "orange",
            "Smearing MTF": "gold",
            "Motion blur MTF": "brown",
            "System MTF": "green"
        }

        # Dibujar cada MTF
        for key, values in Hcuts.items():
            plt.plot(fnAlt_pos, values, label=f"Central ALT position of the {key}",
                     color=colors.get(key, None), linewidth=1.2)

        # Línea de Nyquist en 0.5
        plt.axvline(0.5, color='k', linestyle='--', linewidth=1.0, label="Nyquist Limit")

        # Etiquetas y título
        plt.xlabel("Spatial frequencies f/(1/w) [-]", fontsize=10)
        plt.ylabel("MTF Values", fontsize=10)
        plt.title(f"Central ALT position of the MTFs and Nyquist in band ->{band}", fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.4)

        # Leyenda
        plt.legend(fontsize=7, loc='best')

        # Ajustar límites para que Nyquist quede visible a la derecha
        plt.xlim(0, 0.55)
        plt.ylim(0.35, 1.05)

        # Guardar figura
        fname = os.path.join(directory, f"MTF_ALT_cut_band_{band}.png")
        plt.tight_layout()
        plt.show()
        plt.close()

        self.logger.info(f"Saved MTF plot (ALT only) to {fname}")

        # --- También imprime el valor del MTF en Nyquist ---
        nyq_idx = np.argmin(np.abs(fnAlt_pos - 0.5))
        sys_val = Hcuts["System MTF"][nyq_idx]
        self.logger.info(f"MTF@Nyquist (ALT, band {band}) = {sys_val:.3f}")
        print(f"MTF@Nyquist (ALT, band {band}) = {sys_val:.3f}")